"""Command-line interface for native, local-first research version control."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ai_scientist.protocol.research_vcs import (
    RESEARCH_OBJECT_STATES,
    RESEARCH_RELATION_TYPES,
)
from ai_scientist.utils.privacy import (
    portable_path,
    redact_sensitive_payload,
    redact_sensitive_text,
)

from .research_git import (
    ResearchGitError,
    add_research_object,
    commit_research_stage,
    create_checkpoint,
    create_research_branch,
    create_research_bundle,
    create_research_tag,
    delete_research_branch,
    init_repository,
    list_research_branches,
    list_research_objects,
    list_research_tags,
    load_research_object,
    merge_research_branch,
    preview_research_merge,
    record_research_object,
    rename_research_branch,
    research_blame,
    repository_status,
    reproduce_checkpoint,
    restore_research_bundle,
    restore_research_paths,
    revert_research_checkpoint,
    research_diff,
    research_log,
    research_trajectory,
    research_stage,
    research_unstage,
    show_checkpoint,
    switch_research_branch,
    verify_research_bundle,
    verify_research_repository,
)
from .opportunity_funnel import (
    inspect_opportunity_funnel,
    save_opportunity_allocation,
    save_opportunity_attempt,
    save_opportunity_grade,
    save_opportunity_judgment,
    save_opportunity_pool,
    save_research_direction,
)
from .research_belief import audit_belief_context_projection
from .research_rollout import audit_research_rollout, save_research_rollout


def _print_json(payload: Any) -> None:
    print(
        json.dumps(
            redact_sensitive_payload(payload),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def _display_path(value: Any) -> str:
    return portable_path(str(value), base=Path.cwd())


def _display_text(value: Any) -> str:
    return redact_sensitive_text(str(value))


def _read_question(args: argparse.Namespace) -> str | None:
    if args.question and args.question_file:
        raise ResearchGitError("use only one of --question or --question-file")
    if args.question_file:
        path = Path(args.question_file).expanduser()
        if not path.is_file():
            raise ResearchGitError(f"question file not found: {path}")
        return path.read_text(encoding="utf-8")
    if args.question:
        return f"# Research question\n\n{args.question.strip()}\n"
    return None


def _read_object_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.data is not None and args.file is not None:
        raise ResearchGitError("use only one of --data or --file")
    if args.file is not None:
        path = Path(args.file).expanduser()
        if not path.is_file():
            raise ResearchGitError("research object payload file was not found")
        raw = path.read_text(encoding="utf-8")
    elif args.data is not None:
        raw = args.data
    else:
        raise ResearchGitError("record requires --data or --file")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResearchGitError(
            f"research object payload is invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ResearchGitError("research object payload must be a JSON object")
    return payload


def _parse_relations(values: Sequence[str]) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    for value in values:
        if "=" in value:
            relation_type, separator, remainder = value.partition("=")
        else:
            relation_type, separator, remainder = value.partition(":")
        target, role_separator, role = remainder.partition(":")
        supported = (
            relation_type in RESEARCH_RELATION_TYPES
            or relation_type.startswith(("https://", "http://", "urn:"))
        )
        if not separator or not supported or not target:
            raise ResearchGitError(
                "relation must be TYPE:TARGET[:ROLE]; absolute URI types use "
                "TYPE=TARGET[:ROLE]"
            )
        relation = {"type": relation_type, "target": target}
        if role_separator and role:
            relation["role"] = role
        relations.append(relation)
    return relations


def _parse_assignments(values: Sequence[str], *, label: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw.strip():
            raise ResearchGitError(f"{label} must use NAME=VALUE")
        if name in parsed:
            raise ResearchGitError(f"duplicate {label} name: {name}")
        try:
            parsed[name] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[name] = raw.strip()
    return parsed


def _parse_path_assignments(values: Sequence[str], *, label: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        name, separator, raw = str(value).partition("=")
        name = name.strip()
        path = raw.strip()
        if not separator or not name or not path:
            raise ResearchGitError(f"{label} must use NAME=PATH")
        if name in parsed:
            raise ResearchGitError(f"duplicate {label} name: {name}")
        parsed[name] = path
    return parsed


def _parse_confirmatory_splits(
    digest_values: Sequence[str], file_values: Sequence[str]
) -> dict[str, str]:
    splits = {
        key: str(value).strip()
        for key, value in _parse_assignments(
            digest_values, label="confirmatory split"
        ).items()
    }
    for value in file_values:
        task_id, separator, path = str(value).partition("=")
        task_id = task_id.strip()
        if not separator or not task_id or not path.strip():
            raise ResearchGitError("confirmatory split file must use TASK_ID=PATH")
        if task_id in splits:
            raise ResearchGitError(f"duplicate confirmatory split task: {task_id}")
        splits[task_id] = _hash_local_file(path.strip())
    return splits


def _parse_context_options(
    values: Sequence[str], *, selected: str
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw in values:
        option, separator, reason = str(raw).partition("=")
        option = option.strip()
        if not option:
            raise ResearchGitError("context option requires a name")
        if option != selected and (not separator or not reason.strip()):
            raise ResearchGitError(
                f"rejected context option {option!r} requires OPTION=REASON"
            )
        rows.append(
            {
                "option": option,
                "rejected_because": "" if option == selected else reason.strip(),
            }
        )
    return rows


def _hash_local_file(path_value: str) -> str:
    import hashlib

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ResearchGitError("local file was not found")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResearchGitError("local file could not be read") from exc
    return "sha256:" + digest.hexdigest()


def _read_json_value(path_value: str, *, label: str) -> Any:
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ResearchGitError(f"{label} file was not found")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchGitError(f"{label} file is invalid JSON") from exc


def _read_json_mapping(path_value: str, *, label: str) -> dict[str, Any]:
    value = _read_json_value(path_value, label=label)
    if not isinstance(value, dict):
        raise ResearchGitError(f"{label} must be a JSON object")
    return value


def _saved_object_json(result: Mapping[str, Any]) -> dict[str, Any]:
    recorded = result["object"]
    related = result.get("related") or []
    checkpoint = result.get("checkpoint")
    return {
        "object": recorded.to_dict(),
        "related_objects": [item.to_dict() for item in related],
        "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
    }


def _print_saved_object(
    label: str,
    result: dict[str, Any],
    *,
    as_json: bool,
    guide_repo: str | Path | None = None,
) -> None:
    recorded = result["object"]
    related = result.get("related") or []
    checkpoint = result.get("checkpoint")
    if as_json:
        _print_json(_saved_object_json(result))
        return
    action = "Recorded" if recorded.created else "Reused"
    print(f"{action} {label}: {recorded.object_id} ({recorded.state})")
    for item in related:
        related_action = "Recorded" if item.created else "Reused"
        print(
            f"{related_action} related {item.kind}: " f"{item.object_id} ({item.state})"
        )
    if checkpoint is None:
        print("Checkpoint: not requested")
    elif checkpoint.committed:
        print(f"Checkpoint: {checkpoint.checkpoint_id}")
    else:
        print(f"Checkpoint skipped: {checkpoint.reason}")
    if guide_repo is not None:
        print("Next: xscientist research guide --repo " + shlex.quote(str(guide_repo)))


def _add_scope_arguments(
    parser: argparse.ArgumentParser, *, metric_flag: str = "--scope-metric"
) -> None:
    parser.add_argument("--scope", default="", help="Legacy free-text applicability.")
    parser.add_argument("--population", default="")
    parser.add_argument("--intervention", default="")
    parser.add_argument("--comparator", default="")
    parser.add_argument("--outcome", default="")
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--dataset-slice", action="append", default=[])
    parser.add_argument(metric_flag, dest="scope_metric", default="")
    parser.add_argument("--unit", default="")
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--time-window", default="")
    parser.add_argument("--estimand", default="")


def _scope_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "population": args.population,
        "intervention": args.intervention,
        "comparator": args.comparator,
        "outcome": args.outcome,
        "datasets": args.dataset,
        "dataset_slices": args.dataset_slice,
        "metric": args.scope_metric,
        "unit": args.unit,
        "conditions": args.condition,
        "time_window": args.time_window,
        "estimand": args.estimand,
    }


def _add_program_save_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", default=".")
    parser.add_argument("-m", "--message")
    parser.add_argument("--no-commit", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")


def _build_parser(*, prog: str = "xscientist research") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Version scientific questions, hypotheses, evidence, evaluations, and "
            "manuscripts locally. No server is required and no command pushes automatically."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the local version-control backend and required capabilities.",
    )
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")

    start_parser = subparsers.add_parser(
        "start",
        help="Start a research repository with one plain-language question and falsifiable hypothesis.",
    )
    start_parser.add_argument("path")
    start_parser.add_argument("--question", required=True)
    start_parser.add_argument("--hypothesis", required=True)
    start_parser.add_argument("--falsifier", required=True)
    start_parser.add_argument("--name")
    start_parser.add_argument("--actor", default="human:researcher")
    start_parser.add_argument("--lang", choices=["auto", "en", "zh"], default="auto")
    start_parser.add_argument("--git-user-name")
    start_parser.add_argument("--git-user-email")
    start_parser.add_argument("--json", action="store_true", dest="as_json")

    guide_parser = subparsers.add_parser(
        "guide",
        help="Explain current progress and the next scientific step in plain language.",
    )
    guide_parser.add_argument("--repo", default=".")
    guide_parser.add_argument("--lang", choices=["auto", "en", "zh"], default="auto")
    guide_parser.add_argument("--json", action="store_true", dest="as_json")

    hypothesis_parser = subparsers.add_parser(
        "hypothesis",
        help="Record and checkpoint one falsifiable hypothesis.",
    )
    hypothesis_parser.add_argument("statement")
    hypothesis_parser.add_argument("--falsifier", required=True)
    hypothesis_parser.add_argument("--rationale", default="")
    hypothesis_parser.add_argument("--prediction", action="append", default=[])
    hypothesis_parser.add_argument("--repo", default=".")
    hypothesis_parser.add_argument("-m", "--message")
    hypothesis_parser.add_argument("--no-commit", action="store_true")
    hypothesis_parser.add_argument("--json", action="store_true", dest="as_json")

    plan_parser = subparsers.add_parser(
        "plan",
        help="Record an exploratory plan in plain language before running experiments.",
    )
    plan_parser.add_argument("hypothesis_id")
    plan_parser.add_argument("summary")
    plan_parser.add_argument(
        "--test",
        action="append",
        default=[],
        help="A discriminating or falsification test; repeat as needed.",
    )
    plan_parser.add_argument("--success-rule", default="")
    plan_parser.add_argument("--repo", default=".")
    plan_parser.add_argument("-m", "--message")
    plan_parser.add_argument("--no-commit", action="store_true")
    plan_parser.add_argument("--json", action="store_true", dest="as_json")

    discovery_parser = subparsers.add_parser(
        "discovery",
        help=(
            "Lock a method-discovery contract and distinguish transferable "
            "methods from local engineering gains."
        ),
    )
    discovery_subparsers = discovery_parser.add_subparsers(
        dest="discovery_command", required=True
    )
    discovery_template = discovery_subparsers.add_parser(
        "template",
        help="Print or safely write a complete editable discovery contract.",
    )
    discovery_template.add_argument("--output")
    discovery_template.add_argument("--json", action="store_true", dest="as_json")
    discovery_plan = discovery_subparsers.add_parser(
        "plan",
        help=(
            "Lock target edit scope, strong baselines, multiple conditions, "
            "resource limits, and sealed feedback."
        ),
    )
    discovery_plan.add_argument("hypothesis_id")
    discovery_plan.add_argument(
        "spec",
        help="JSON method-discovery contract; see docs/METHOD_DISCOVERY_PROTOCOL.md.",
    )
    discovery_plan.add_argument(
        "--context",
        help="Complete context snapshot visible when selecting this mechanism and design.",
    )
    discovery_plan.add_argument("--repo", default=".")
    discovery_plan.add_argument("-m", "--message")
    discovery_plan.add_argument("--no-commit", action="store_true")
    discovery_plan.add_argument("--json", action="store_true", dest="as_json")

    discovery_assess = discovery_subparsers.add_parser(
        "assess",
        help=(
            "Score a committed candidate on every condition and record the "
            "generalization verdict."
        ),
    )
    discovery_assess.add_argument("contract_id")
    discovery_assess.add_argument(
        "results", help="JSON candidate and per-condition measurements."
    )
    discovery_assess.add_argument(
        "--evidence",
        action="append",
        required=True,
        help="Condition evidence selector.",
    )
    discovery_assess.add_argument("--repo", default=".")
    discovery_assess.add_argument("-m", "--message")
    discovery_assess.add_argument("--no-commit", action="store_true")
    discovery_assess.add_argument("--json", action="store_true", dest="as_json")

    program_parser = subparsers.add_parser(
        "program",
        help="Run the competitive-hypothesis, information-value, and theory-depth loop.",
    )
    program_subparsers = program_parser.add_subparsers(
        dest="program_command", required=True
    )
    program_template = program_subparsers.add_parser(
        "template", help="Print or write editable deep-research JSON templates."
    )
    program_template.add_argument("--output")
    program_template.add_argument("--json", action="store_true", dest="as_json")

    program_portfolio = program_subparsers.add_parser(
        "portfolio", help="Lock primary, alternative, and optional null hypotheses."
    )
    program_portfolio.add_argument("primary_id")
    program_portfolio.add_argument("--alternative", action="append", required=True)
    program_portfolio.add_argument("--null", dest="null_id")
    program_portfolio.add_argument("--question", required=True)
    program_portfolio.add_argument(
        "--prior", action="append", default=[], help="Hypothesis selector=weight."
    )
    _add_program_save_arguments(program_portfolio)

    program_prediction = program_subparsers.add_parser(
        "prediction", help="Lock an outcome that distinguishes competing hypotheses."
    )
    program_prediction.add_argument("portfolio_id")
    program_prediction.add_argument("hypothesis_id")
    program_prediction.add_argument("--when", required=True)
    program_prediction.add_argument("--expect", required=True)
    program_prediction.add_argument("--distinguishes", action="append", required=True)
    program_prediction.add_argument("--falsifier", required=True)
    _add_program_save_arguments(program_prediction)

    program_priority = program_subparsers.add_parser(
        "prioritize", help="Rank candidate experiments by expected information value."
    )
    program_priority.add_argument("portfolio_id")
    program_priority.add_argument(
        "candidates", help="JSON array or object with experiment_candidates."
    )
    _add_program_save_arguments(program_priority)

    program_posterior = program_subparsers.add_parser(
        "posterior",
        help="Bind a selected attempt and observation to a Bayesian portfolio update.",
    )
    program_posterior.add_argument("portfolio_id")
    program_posterior.add_argument("priority_id")
    program_posterior.add_argument("attempt_id")
    program_posterior.add_argument("evidence_id")
    program_posterior.add_argument("--observed", required=True)
    program_posterior.add_argument(
        "--likelihood",
        action="append",
        required=True,
        help="Hypothesis selector=likelihood (0..1).",
    )
    _add_program_save_arguments(program_posterior)

    program_mechanism = program_subparsers.add_parser(
        "mechanism", help="Record an intervention-testable causal mechanism."
    )
    program_mechanism.add_argument("hypothesis_id")
    program_mechanism.add_argument("statement")
    program_mechanism.add_argument("--mediator", action="append", required=True)
    program_mechanism.add_argument("--intervention", action="append", required=True)
    program_mechanism.add_argument("--rival", action="append", default=[])
    program_mechanism.add_argument("--evidence", action="append", default=[])
    program_mechanism.add_argument(
        "--status",
        choices=["proposed", "tested", "validated", "refuted"],
        default="proposed",
    )
    _add_program_save_arguments(program_mechanism)

    program_quality = program_subparsers.add_parser(
        "quality", help="Assess evidence quality and bias across fixed domains."
    )
    program_quality.add_argument("evidence_id")
    program_quality.add_argument("assessment", help="JSON object with domains/notes.")
    program_quality.add_argument("--assessor", required=True)
    program_quality.add_argument("--independent", action="store_true")
    _add_program_save_arguments(program_quality)

    program_boundary = program_subparsers.add_parser(
        "boundary", help="Map claim applicability and held-out transfer conditions."
    )
    program_boundary.add_argument("claim_id")
    program_boundary.add_argument(
        "matrix", help="JSON array or object with boundary_rows."
    )
    _add_program_save_arguments(program_boundary)

    program_review = program_subparsers.add_parser(
        "review", help="Inspect or checkpoint structural gaps and anomalies."
    )
    program_review.add_argument("--record", action="store_true")
    _add_program_save_arguments(program_review)

    program_followup = program_subparsers.add_parser(
        "followup",
        help="Queue a finite set of actions from the latest strategy review gaps.",
    )
    program_followup.add_argument(
        "--review", help="Optional recorded research_review object to bind."
    )
    program_followup.add_argument("--max-actions", type=int, default=1)
    _add_program_save_arguments(program_followup)

    program_claim = program_subparsers.add_parser(
        "claim", help="Explain one claim's support, refutation, mechanism, and gaps."
    )
    program_claim.add_argument("claim_id")
    program_claim.add_argument("--repo", default=".")
    program_claim.add_argument("--json", action="store_true", dest="as_json")

    opportunity_parser = subparsers.add_parser(
        "opportunity",
        help=(
            "Record a FAR-inspired research-direction → opportunity-pool → "
            "attempt → independent judge/grade funnel."
        ),
    )
    opportunity_subparsers = opportunity_parser.add_subparsers(
        dest="opportunity_command", required=True
    )
    opportunity_direction = opportunity_subparsers.add_parser(
        "direction", help="Lock a research direction before extracting opportunities."
    )
    opportunity_direction.add_argument("direction_id")
    opportunity_direction.add_argument("statement")
    opportunity_direction.add_argument("objective")
    opportunity_direction.add_argument("--domain", default="")
    opportunity_direction.add_argument("--success-definition", default="")
    opportunity_direction.add_argument("--constraint", action="append", default=[])
    opportunity_direction.add_argument("--source-ref", action="append", default=[])
    _add_program_save_arguments(opportunity_direction)

    opportunity_pool = opportunity_subparsers.add_parser(
        "pool", help="Persist a complete candidate JSON array from a bounded extractor."
    )
    opportunity_pool.add_argument("direction_id")
    opportunity_pool.add_argument(
        "candidates", help="JSON array or file containing candidates."
    )
    opportunity_pool.add_argument("--incomplete", action="store_true")
    opportunity_pool.add_argument("--extraction-notes", default="")
    _add_program_save_arguments(opportunity_pool)

    opportunity_attempt = opportunity_subparsers.add_parser(
        "attempt", help="Record an explicit KNOWN/NEW/FIX/NONE opportunity outcome."
    )
    opportunity_attempt.add_argument("pool_id")
    opportunity_attempt.add_argument("candidate_id")
    opportunity_attempt.add_argument("outcome", choices=["known", "new", "fix", "none"])
    opportunity_attempt.add_argument("summary")
    opportunity_attempt.add_argument("--evidence-ref", action="append", default=[])
    opportunity_attempt.add_argument(
        "--evidence-object-id", action="append", default=[]
    )
    opportunity_attempt.add_argument("--runner", default="")
    _add_program_save_arguments(opportunity_attempt)

    opportunity_judge = opportunity_subparsers.add_parser(
        "judge", help="Record a provenance-disjoint judgment for an attempt."
    )
    opportunity_judge.add_argument("attempt_id")
    opportunity_judge.add_argument("verdict", choices=["pass", "fail", "known"])
    opportunity_judge.add_argument("evaluator_id")
    opportunity_judge.add_argument("summary")
    opportunity_judge.add_argument("--evidence-ref", action="append", default=[])
    opportunity_judge.add_argument("--evidence-object-id", action="append", default=[])
    opportunity_judge.add_argument(
        "--allow-stage-override",
        action="store_true",
        help="Allow a non-NEW retrospective judgment; requires --override-reason.",
    )
    opportunity_judge.add_argument("--override-reason", default="")
    _add_program_save_arguments(opportunity_judge)

    opportunity_grade = opportunity_subparsers.add_parser(
        "grade", help="Record an independent known/minor/substantial grade."
    )
    opportunity_grade.add_argument("judgment_id")
    opportunity_grade.add_argument("grade", choices=["known", "minor", "substantial"])
    opportunity_grade.add_argument("evaluator_id")
    opportunity_grade.add_argument("summary")
    opportunity_grade.add_argument("--evidence-ref", action="append", default=[])
    opportunity_grade.add_argument("--evidence-object-id", action="append", default=[])
    opportunity_grade.add_argument(
        "--allow-stage-override",
        action="store_true",
        help="Allow grading a non-PASS/KNOWN judgment; requires --override-reason.",
    )
    opportunity_grade.add_argument("--override-reason", default="")
    _add_program_save_arguments(opportunity_grade)

    opportunity_allocate = opportunity_subparsers.add_parser(
        "allocate",
        help=(
            "Rank declared expected yields; missing success probabilities stay "
            "ineligible and any neutral artifact-factor assumption is recorded."
        ),
    )
    opportunity_allocate.add_argument("pool_id")
    opportunity_allocate.add_argument(
        "--objective",
        choices=["artifact_yield", "importance_yield", "best_artifact"],
        default="artifact_yield",
    )
    opportunity_allocate.add_argument("--max-attempts", type=int)
    opportunity_allocate.add_argument(
        "--calibration-status", default="declared_inputs_not_calibrated"
    )
    opportunity_allocate.add_argument(
        "--probability-semantics",
        choices=[
            "conditional_artifact_given_success",
            "joint_artifact_probability",
        ],
        default="conditional_artifact_given_success",
    )
    _add_program_save_arguments(opportunity_allocate)

    opportunity_inspect = opportunity_subparsers.add_parser(
        "inspect", help="Show funnel coverage and unattempted/orphan rows."
    )
    opportunity_inspect.add_argument("pool_id")
    opportunity_inspect.add_argument("--repo", default=".")
    opportunity_inspect.add_argument("--json", action="store_true", dest="as_json")

    rollout_parser = subparsers.add_parser(
        "rollout",
        help=(
            "Record a metadata-only research-policy/tool rollout with a task rubric, "
            "turn credit trace, and observational evaluator summary."
        ),
    )
    rollout_parser.add_argument(
        "episode",
        help="JSON file containing task_hash, budget, tool calls, turns, and evaluations.",
    )
    _add_program_save_arguments(rollout_parser)

    rollout_audit_parser = subparsers.add_parser(
        "rollout-audit",
        help=(
            "Audit a saved metadata-only rollout without exposing payloads; "
            "completed reports fail closed without an evidence resolver."
        ),
    )
    rollout_audit_parser.add_argument(
        "report",
        help="JSON rollout payload or JSON output from `research rollout --json`.",
    )
    rollout_audit_parser.add_argument(
        "--evidence-hash",
        action="append",
        default=[],
        help="Content hash known to the local evidence index (repeatable).",
    )
    rollout_audit_parser.add_argument(
        "--trust-store",
        help=(
            "Local JSON trust store keyed by attestation key_id; required to "
            "verify an independent evaluator signature."
        ),
    )
    rollout_audit_parser.add_argument(
        "--max-attestation-age-seconds",
        type=int,
        help="Optional non-negative evaluator-attestation freshness limit.",
    )
    rollout_audit_parser.add_argument("--json", action="store_true", dest="as_json")

    verifier_authority_parser = subparsers.add_parser(
        "verifier-authority",
        help=(
            "Prepare, finalize, or verify the external Ed25519 authority "
            "receipt required by top-venue publication gates."
        ),
    )
    verifier_authority_subparsers = verifier_authority_parser.add_subparsers(
        dest="verifier_authority_command", required=True
    )
    verifier_authority_prepare = verifier_authority_subparsers.add_parser(
        "prepare",
        help="Build the exact hash-only payload an external verifier must sign.",
    )
    verifier_authority_prepare.add_argument("--paper-dir", default=".")
    verifier_authority_prepare.add_argument("--identity", required=True)
    verifier_authority_prepare.add_argument("--output", required=True)
    verifier_authority_prepare.add_argument("--force", action="store_true")
    verifier_authority_prepare.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    verifier_authority_finalize = verifier_authority_subparsers.add_parser(
        "finalize",
        help="Bind an external Ed25519 attestation to the current report.",
    )
    verifier_authority_finalize.add_argument("--paper-dir", default=".")
    verifier_authority_finalize.add_argument("--identity", required=True)
    verifier_authority_finalize.add_argument("--attestation", required=True)
    verifier_authority_finalize.add_argument("--output")
    verifier_authority_finalize.add_argument("--force", action="store_true")
    verifier_authority_finalize.add_argument(
        "--json", action="store_true", dest="as_json"
    )
    verifier_authority_verify = verifier_authority_subparsers.add_parser(
        "verify",
        help=(
            "Verify only the receipt/signature binding against an explicit "
            "external trust store; submission readiness remains unknown until "
            "the complete evidence gate runs."
        ),
    )
    verifier_authority_verify.add_argument("--paper-dir", default=".")
    verifier_authority_verify.add_argument("--receipt")
    verifier_authority_verify.add_argument("--trust-store", required=True)
    verifier_authority_verify.add_argument(
        "--json", action="store_true", dest="as_json"
    )

    belief_parser = subparsers.add_parser(
        "belief",
        help=(
            "Build a bounded candidate-belief projection from the immutable "
            "Research VCS source closure."
        ),
    )
    belief_parser.add_argument("target", nargs="+", help="Research Object selector.")
    belief_parser.add_argument("--ref", default="WORKTREE")
    belief_parser.add_argument(
        "--as-of",
        help=(
            "Timezone-aware ISO-8601 validity boundary; defaults to the latest "
            "source timestamp for deterministic replay."
        ),
    )
    belief_parser.add_argument("--budget", type=int, default=4000)
    belief_parser.add_argument("--repo", default=".")
    belief_parser.add_argument("--json", action="store_true", dest="as_json")

    belief_audit_parser = subparsers.add_parser(
        "belief-audit",
        help="Audit a belief-context projection without returning source payloads.",
    )
    belief_audit_parser.add_argument(
        "report",
        help="JSON projection, research context, or JSON output containing context.",
    )
    belief_audit_parser.add_argument("--json", action="store_true", dest="as_json")

    literature_parser = subparsers.add_parser(
        "literature",
        help="Record a reproducible literature search and source-qualified passages.",
    )
    literature_subparsers = literature_parser.add_subparsers(
        dest="literature_command", required=True
    )
    literature_plan = literature_subparsers.add_parser(
        "plan", help="Lock queries and selection criteria before searching."
    )
    literature_plan.add_argument("question")
    literature_plan.add_argument("--query", action="append", required=True)
    literature_plan.add_argument("--provider", action="append", default=[])
    literature_plan.add_argument("--include", action="append", default=[])
    literature_plan.add_argument("--exclude", action="append", default=[])
    literature_plan.add_argument("--repo", default=".")
    literature_plan.add_argument("-m", "--message")
    literature_plan.add_argument("--no-commit", action="store_true")
    literature_plan.add_argument("--json", action="store_true", dest="as_json")

    literature_receipt = literature_subparsers.add_parser(
        "receipt", help="Record every ranked candidate returned by one provider call."
    )
    literature_receipt.add_argument("plan_id")
    literature_receipt.add_argument("--provider", required=True)
    literature_receipt.add_argument("--query", required=True)
    literature_receipt.add_argument(
        "--results",
        required=True,
        help="JSON array, or object with a candidates array; secrets and raw bodies are removed.",
    )
    literature_receipt.add_argument("--retrieved-at", default="")
    literature_receipt.add_argument("--corpus-version", default="")
    literature_receipt.add_argument("--corpus-snapshot-hash", default="")
    literature_receipt.add_argument("--query-rewrite", action="append", default=[])
    literature_receipt.add_argument(
        "--filter", action="append", default=[], help="Search filter as NAME=VALUE."
    )
    literature_receipt.add_argument("--retriever", default="")
    literature_receipt.add_argument("--embedding-model", default="")
    literature_receipt.add_argument("--reranker", default="")
    literature_receipt.add_argument("--cursor", default="")
    literature_receipt.add_argument("--page", type=int)
    literature_receipt.add_argument(
        "--incomplete",
        action="store_true",
        help="Declare that pagination or provider limits prevented a complete candidate set.",
    )
    literature_receipt.add_argument("--error", action="append", default=[])
    literature_receipt.add_argument("--repo", default=".")
    literature_receipt.add_argument("-m", "--message")
    literature_receipt.add_argument("--no-commit", action="store_true")
    literature_receipt.add_argument("--json", action="store_true", dest="as_json")

    literature_source = literature_subparsers.add_parser(
        "source", help="Freeze one selected source by identifier and content hash."
    )
    literature_source.add_argument("receipt_id")
    literature_source.add_argument("title")
    source_identity = literature_source.add_mutually_exclusive_group(required=True)
    source_identity.add_argument("--content-hash")
    source_identity.add_argument(
        "--file", help="Hash a local source without storing its path."
    )
    literature_source.add_argument("--metadata-hash")
    literature_source.add_argument("--doi", default="")
    literature_source.add_argument("--pmid", default="")
    literature_source.add_argument("--arxiv-id", default="")
    literature_source.add_argument("--url", default="")
    literature_source.add_argument("--license", default="", dest="license_name")
    literature_source.add_argument("--retraction-status", default="unknown")
    literature_source.add_argument("--status-provider", default="")
    literature_source.add_argument("--status-checked-at", default="")
    literature_source.add_argument("--status-notice-id", default="")
    literature_source.add_argument("--supersedes", dest="previous_source_id")
    literature_source.add_argument("--repo", default=".")
    literature_source.add_argument("-m", "--message")
    literature_source.add_argument("--no-commit", action="store_true")
    literature_source.add_argument("--json", action="store_true", dest="as_json")

    literature_update = literature_subparsers.add_parser(
        "update", help="Append an immutable correction, retraction, or status check."
    )
    literature_update.add_argument("source_id")
    literature_update.add_argument("--status", required=True)
    literature_update.add_argument("--provider", required=True)
    literature_update.add_argument("--checked-at", required=True)
    literature_update.add_argument("--type", default="status_check", dest="update_type")
    literature_update.add_argument("--notice-id", default="")
    literature_update.add_argument("--detail", default="")
    literature_update.add_argument("--repo", default=".")
    literature_update.add_argument("-m", "--message")
    literature_update.add_argument("--no-commit", action="store_true")
    literature_update.add_argument("--json", action="store_true", dest="as_json")

    literature_passage = literature_subparsers.add_parser(
        "passage",
        help="Record an exact quote, locator, source, and semantic direction.",
    )
    literature_passage.add_argument("source_id")
    literature_passage.add_argument("quote")
    literature_passage.add_argument("--locator", required=True)
    literature_passage.add_argument("--prefix", default="")
    literature_passage.add_argument("--suffix", default="")
    literature_passage.add_argument("--start", type=int)
    literature_passage.add_argument("--end", type=int)
    literature_passage.add_argument("--supports", action="append", default=[])
    literature_passage.add_argument("--refutes", action="append", default=[])
    literature_passage.add_argument("--context")
    _add_scope_arguments(literature_passage, metric_flag="--metric")
    literature_passage.add_argument("--repo", default=".")
    literature_passage.add_argument("-m", "--message")
    literature_passage.add_argument("--no-commit", action="store_true")
    literature_passage.add_argument("--json", action="store_true", dest="as_json")

    preregistration_parser = subparsers.add_parser(
        "preregister",
        help="Lock a confirmatory plan and dataset split before experiments run.",
    )
    preregistration_parser.add_argument("hypothesis_id")
    preregistration_parser.add_argument("--dataset", required=True)
    preregistration_parser.add_argument("--metric", required=True)
    preregistration_parser.add_argument("--baseline", required=True)
    split_source = preregistration_parser.add_mutually_exclusive_group(required=True)
    split_source.add_argument(
        "--split-hash",
        help="Frozen dataset split digest as sha256:<64 hexadecimal characters>.",
    )
    split_source.add_argument(
        "--split-file",
        help="Hash a local dataset split without storing its path or contents.",
    )
    preregistration_parser.add_argument("--registered-by", required=True)
    preregistration_parser.add_argument("--minimum-effect", type=float)
    preregistration_parser.add_argument("--alpha", type=float, default=0.05)
    preregistration_parser.add_argument("--minimum-seeds", type=int, default=3)
    preregistration_parser.add_argument("--repo", default=".")
    preregistration_parser.add_argument("-m", "--message")
    preregistration_parser.add_argument("--no-commit", action="store_true")
    preregistration_parser.add_argument("--json", action="store_true", dest="as_json")

    confirm_parser = subparsers.add_parser(
        "confirm",
        help=(
            "Host-lock a generated multi-task paper plan, empirical snapshot, "
            "splits, and confirmatory execution queue."
        ),
    )
    confirm_parser.add_argument("--paper-dir", default=".")
    confirm_parser.add_argument(
        "--registered-by",
        default="recorder:xscientist-user",
        help=(
            "Self-reported recorder identity for provenance; this does not grant "
            "human or independent-verifier authority."
        ),
    )
    confirm_parser.add_argument(
        "--split",
        action="append",
        default=[],
        metavar="TASK_ID=SHA256",
        help="Frozen split digest for one planned task; repeat for every task.",
    )
    confirm_parser.add_argument(
        "--split-file",
        action="append",
        default=[],
        metavar="TASK_ID=PATH",
        help="Hash a local split definition for one task without storing its path.",
    )
    confirm_parser.add_argument(
        "--data-manifest-hash",
        help="Optional expected hash; must equal the host-verified data manifest.",
    )
    confirm_parser.add_argument(
        "--data-snapshot-id",
        help="Optional expected ID; must equal the host-verified read-only snapshot.",
    )
    confirm_parser.add_argument("-m", "--message")
    confirm_parser.add_argument("--json", action="store_true", dest="as_json")

    trajectory_bind_parser = subparsers.add_parser(
        "trajectory-bind",
        help=(
            "Bind one immutable experiment-registry row to its typed Research VCS "
            "attempt and origin checkpoint."
        ),
    )
    trajectory_bind_parser.add_argument("--paper-dir", default=".")
    trajectory_bind_parser.add_argument("--record-id", required=True)
    trajectory_bind_parser.add_argument("--attempt", required=True)
    trajectory_bind_parser.add_argument("-m", "--message")
    trajectory_bind_parser.add_argument("--json", action="store_true", dest="as_json")

    disposition_parser = subparsers.add_parser(
        "attempt-disposition",
        help=(
            "Preserve a failed/timed-out/cancelled attempt and append its explicit "
            "auditable disposition; the host independently decides whether it "
            "resolves a publication blocker."
        ),
    )
    disposition_parser.add_argument("--paper-dir", default=".")
    disposition_parser.add_argument("--record-id", required=True)
    disposition_parser.add_argument(
        "--disposition",
        required=True,
        choices=[
            "terminal_negative",
            "technical_failure_retried",
            "approved_deviation",
            "excluded_with_reason",
        ],
        help=(
            "Typed audit outcome. Only terminal_negative and a valid "
            "technical_failure_retried resolve the publication blocker."
        ),
    )
    disposition_parser.add_argument("--reason", required=True)
    disposition_parser.add_argument("--retry-record-id")
    disposition_parser.add_argument(
        "--approved-before-unblinding",
        action="store_true",
        help=(
            "Record the caller's timing assertion for audit only; it does not "
            "establish independent approval or publication authority."
        ),
    )
    disposition_parser.add_argument(
        "--negative-result-artifact",
        metavar="PATH",
        help=(
            "Repository-contained result file for terminal_negative. The host "
            "reads it as a bounded regular file and recomputes its SHA-256."
        ),
    )
    disposition_parser.add_argument(
        "--negative-result-evidence",
        metavar="OBJECT",
        help=(
            "Evidence object derived only from the failed attempt, containing a "
            "hash-valid metric assessment of the negative result."
        ),
    )
    disposition_parser.add_argument(
        "--recorded-by",
        default="recorder:xscientist-user",
        help=(
            "Self-reported recorder for this disposition; publication resolution "
            "is recomputed independently by the host attestor."
        ),
    )
    disposition_parser.add_argument("-m", "--message")
    disposition_parser.add_argument("--json", action="store_true", dest="as_json")

    experiment_parser = subparsers.add_parser(
        "experiment",
        help=(
            "Record one terminal successful, failed, timed-out, cancelled, "
            "rejected, or orphaned experiment."
        ),
    )
    experiment_parser.add_argument("summary")
    experiment_parser.add_argument(
        "--status",
        required=True,
        choices=[
            "success",
            "completed",
            "failed",
            "error",
            "timeout",
            "timed_out",
            "cancelled",
            "canceled",
            "rejected",
            "orphan",
            "orphaned",
        ],
    )
    experiment_parser.add_argument(
        "--study-phase",
        choices=["exploratory", "confirmatory"],
        default="exploratory",
    )
    experiment_parser.add_argument(
        "--task",
        help="Locked research-plan task id; required by multi-task confirmation.",
    )
    experiment_parser.add_argument("--plan")
    experiment_parser.add_argument(
        "--priority", help="Locked priority that selected an experiment design."
    )
    experiment_parser.add_argument("--preregistration")
    experiment_parser.add_argument("--intervention", action="append", default=[])
    experiment_parser.add_argument("--boundary-condition", default="")
    experiment_parser.add_argument(
        "--boundary-role",
        choices=["development", "transfer", "heldout", "scale"],
        default="",
    )
    experiment_parser.add_argument(
        "--metric", action="append", default=[], help="Metric as NAME=VALUE."
    )
    experiment_parser.add_argument(
        "--config",
        action="append",
        default=[],
        help="Exact executed configuration as NAME=VALUE (repeatable).",
    )
    experiment_parser.add_argument(
        "--producer-id",
        help="Actor/service that actually produced this result.",
    )
    experiment_parser.add_argument(
        "--result-artifact",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Ingest one durable result artifact into Research VCS CAS "
            "(repeatable; required for a completed campaign task)."
        ),
    )
    experiment_parser.add_argument("--seed", action="append", type=int, default=[])
    experiment_parser.add_argument("--environment-hash")
    experiment_parser.add_argument(
        "--dependency-lock-hash", action="append", default=[]
    )
    experiment_parser.add_argument(
        "--dependency-lock-file", action="append", default=[]
    )
    experiment_parser.add_argument("--dataset-hash", action="append", default=[])
    experiment_parser.add_argument("--code-commit")
    experiment_parser.add_argument("--failure-class", default="")
    experiment_parser.add_argument(
        "--reproduce-command",
        help="Shell-free command stored for later `research reproduce --execute`.",
    )
    experiment_parser.add_argument("--repo", default=".")
    experiment_parser.add_argument("-m", "--message")
    experiment_parser.add_argument("--no-commit", action="store_true")
    experiment_parser.add_argument("--json", action="store_true", dest="as_json")

    evidence_parser = subparsers.add_parser(
        "evidence",
        help="Bind a result to experiment attempts and supported/refuted objects.",
    )
    evidence_parser.add_argument("result")
    evidence_parser.add_argument("--attempt", action="append", required=True)
    evidence_parser.add_argument("--supports", action="append", default=[])
    evidence_parser.add_argument("--refutes", action="append", default=[])
    evidence_parser.add_argument(
        "--metric", action="append", default=[], help="Metric as NAME=VALUE."
    )
    evidence_parser.add_argument(
        "--reproduce-command",
        help="Shell-free command inherited from the bound experiment checkpoint.",
    )
    evidence_parser.add_argument("--verified", action="store_true")
    _add_scope_arguments(evidence_parser)
    evidence_parser.add_argument(
        "--verifier",
        help="Independent verifier identity; required with --verified.",
    )
    evidence_parser.add_argument("--repo", default=".")
    evidence_parser.add_argument("-m", "--message")
    evidence_parser.add_argument("--no-commit", action="store_true")
    evidence_parser.add_argument("--json", action="store_true", dest="as_json")

    estimand_parser = subparsers.add_parser(
        "estimand",
        help="Define the population, outcome, treatment contrast, and target measure.",
    )
    estimand_parser.add_argument("outcome")
    estimand_parser.add_argument("--population", required=True)
    estimand_parser.add_argument("--intervention", default="")
    estimand_parser.add_argument("--comparator", default="")
    estimand_parser.add_argument("--time-window", default="")
    estimand_parser.add_argument("--summary-measure", default="")
    estimand_parser.add_argument("--repo", default=".")
    estimand_parser.add_argument("-m", "--message")
    estimand_parser.add_argument("--no-commit", action="store_true")
    estimand_parser.add_argument("--json", action="store_true", dest="as_json")

    effect_parser = subparsers.add_parser(
        "effect", help="Record an effect estimate with explicit uncertainty."
    )
    effect_parser.add_argument("estimand_id")
    effect_parser.add_argument("estimate", type=float)
    effect_parser.add_argument("--metric", required=True)
    effect_parser.add_argument("--unit", default="")
    effect_parser.add_argument("--confidence-level", type=float, default=0.95)
    effect_parser.add_argument("--lower", type=float)
    effect_parser.add_argument("--upper", type=float)
    effect_parser.add_argument("--standard-error", type=float)
    effect_parser.add_argument(
        "--from", action="append", default=[], dest="derived_from"
    )
    effect_parser.add_argument("--repo", default=".")
    effect_parser.add_argument("-m", "--message")
    effect_parser.add_argument("--no-commit", action="store_true")
    effect_parser.add_argument("--json", action="store_true", dest="as_json")

    infer_parser = subparsers.add_parser(
        "infer",
        help="Record the explicit reasoning from evidence premises to a conclusion.",
    )
    infer_parser.add_argument("statement")
    infer_parser.add_argument("--premise", action="append", required=True)
    infer_parser.add_argument("--warrant", required=True)
    infer_parser.add_argument("--method", action="append", default=[])
    infer_parser.add_argument("--assumption", action="append", default=[])
    infer_parser.add_argument("--context")
    infer_parser.add_argument("--repo", default=".")
    infer_parser.add_argument("-m", "--message")
    infer_parser.add_argument("--no-commit", action="store_true")
    infer_parser.add_argument("--json", action="store_true", dest="as_json")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest one schema-bound JSON evidence receipt from an external tool.",
    )
    ingest_parser.add_argument("file")
    ingest_parser.add_argument("--attempt", action="append", required=True)
    ingest_parser.add_argument("--supports", action="append", default=[])
    ingest_parser.add_argument("--refutes", action="append", default=[])
    ingest_parser.add_argument("--repo", default=".")
    ingest_parser.add_argument("-m", "--message")
    ingest_parser.add_argument("--no-commit", action="store_true")
    ingest_parser.add_argument("--json", action="store_true", dest="as_json")

    review_parser = subparsers.add_parser(
        "review",
        help=(
            "Record a local advisory review. Declared local identities cannot grant "
            "external publication authority or promote a verified claim."
        ),
    )
    review_parser.add_argument("summary")
    review_parser.add_argument("--evaluates", action="append", required=True)
    review_parser.add_argument("--verifier", required=True)
    review_parser.add_argument(
        "--decision",
        choices=["pass", "hold"],
        required=True,
        help=(
            "Local advisory recommendation. 'pass' is retained as requested but "
            "cannot promote a claim or grant publication authority."
        ),
    )
    review_parser.add_argument("--failure", action="append", default=[])
    review_parser.add_argument("--repo", default=".")
    review_parser.add_argument("-m", "--message")
    review_parser.add_argument("--no-commit", action="store_true")
    review_parser.add_argument("--json", action="store_true", dest="as_json")

    claim_parser = subparsers.add_parser(
        "claim",
        help="Record an evidence-bound claim; verified claims require a passing gate.",
    )
    claim_parser.add_argument("statement")
    claim_parser.add_argument("--evidence", action="append", required=True)
    _add_scope_arguments(claim_parser, metric_flag="--metric")
    claim_parser.add_argument("--gate")
    claim_parser.add_argument("--verified", action="store_true")
    claim_parser.add_argument(
        "--contribution-level",
        choices=["execution", "engineering_optimization", "method_discovery"],
        default="",
        help="Declare how strong the scientific contribution is intended to be.",
    )
    claim_parser.add_argument(
        "--depth-level",
        choices=["descriptive", "causal", "transferable"],
        default="descriptive",
        help="Opt into mechanism, quality, and transfer promotion gates.",
    )
    claim_parser.add_argument("--mechanism", action="append", default=[])
    claim_parser.add_argument("--quality", action="append", default=[])
    claim_parser.add_argument("--transfer", action="append", default=[])
    claim_parser.add_argument("--repo", default=".")
    claim_parser.add_argument("-m", "--message")
    claim_parser.add_argument("--no-commit", action="store_true")
    claim_parser.add_argument("--json", action="store_true", dest="as_json")

    init_parser = subparsers.add_parser(
        "init", help="Initialize a local research repository."
    )
    init_parser.add_argument("path")
    init_parser.add_argument("--name")
    init_parser.add_argument("--question")
    init_parser.add_argument("--question-file")
    init_parser.add_argument(
        "--policy",
        choices=["manual", "stage", "milestone"],
        default="milestone",
    )
    init_parser.add_argument("--actor", default="xscientist")
    init_parser.add_argument("--git-user-name")
    init_parser.add_argument("--git-user-email")
    init_parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    init_parser.add_argument("--no-commit", action="store_true")
    init_parser.add_argument("--json", action="store_true", dest="as_json")

    status_parser = subparsers.add_parser(
        "status", help="Show research and storage status."
    )
    status_parser.add_argument("--repo", default=".")
    status_parser.add_argument("--json", action="store_true", dest="as_json")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit claim-to-evidence-to-reproduction closure without disclosing payloads.",
    )
    audit_parser.add_argument("ref", nargs="?", default="HEAD")
    audit_parser.add_argument("--repo", default=".")
    audit_parser.add_argument(
        "--level", choices=["trace", "replay", "verify"], default="trace"
    )
    audit_parser.add_argument("--no-objects", action="store_true")
    audit_parser.add_argument("--json", action="store_true", dest="as_json")

    context_parser = subparsers.add_parser(
        "context",
        help="Inspect or record the exact evidence and memory visible to a decision.",
    )
    context_parser.add_argument("target", nargs="+", help="Research Object selector.")
    context_parser.add_argument(
        "--intent",
        choices=["decide", "continue", "write", "audit", "reproduce"],
        default="decide",
    )
    context_parser.add_argument("--decision-kind", default="research_decision")
    context_parser.add_argument("--selected", default="")
    context_parser.add_argument(
        "--option",
        action="append",
        default=[],
        help="Decision option; rejected options use OPTION=REJECTION_REASON.",
    )
    context_parser.add_argument("--rationale", action="append", default=[])
    context_parser.add_argument("--constraint", action="append", default=[])
    context_parser.add_argument("--memory-ref", action="append", default=[])
    context_parser.add_argument("--ref", default="WORKTREE")
    context_parser.add_argument(
        "--belief-as-of",
        help="Timezone-aware ISO-8601 validity boundary for candidate beliefs.",
    )
    context_parser.add_argument("--budget", type=int, default=4000)
    context_parser.add_argument("--record", action="store_true")
    context_parser.add_argument("--no-commit", action="store_true")
    context_parser.add_argument("--repo", default=".")
    context_output = context_parser.add_mutually_exclusive_group()
    context_output.add_argument("--json", action="store_true", dest="as_json")
    context_output.add_argument(
        "--prompt",
        action="store_true",
        help="Print only the bounded, source-bound working memory for an agent.",
    )

    decide_parser = subparsers.add_parser(
        "decide",
        help="Explain whether the next research transition should checkpoint, fork, or merge.",
    )
    decide_parser.add_argument(
        "event",
        choices=[
            "observation",
            "hypothesis",
            "preregistration",
            "experiment-started",
            "experiment-completed",
            "experiment-failed",
            "evidence",
            "review",
            "gate",
            "manuscript",
            "release",
            "method-change",
            "contradiction",
            "replication",
            "agent-candidate",
            "merge-candidate",
        ],
    )
    decide_parser.add_argument("--repo", default=".")
    decide_parser.add_argument("--name", default="")
    decide_parser.add_argument("--state", default="")
    decide_parser.add_argument("--source-branch")
    decide_parser.add_argument("--competing-hypothesis", action="store_true")
    decide_parser.add_argument("--contradictory-evidence", action="store_true")
    decide_parser.add_argument("--protocol-change", action="store_true")
    decide_parser.add_argument("--independent-replication", action="store_true")
    decide_parser.add_argument(
        "--record",
        action="store_true",
        help="Adopt the preview as a context-bound Research VCS decision object.",
    )
    decide_parser.add_argument(
        "--actor",
        default="research-transition-policy",
        help="Recorder identity for an adopted deterministic policy decision.",
    )
    decide_parser.add_argument("--no-commit", action="store_true")
    decide_parser.add_argument("--json", action="store_true", dest="as_json")

    tree_parser = subparsers.add_parser(
        "tree",
        help="Show the payload-free semantic technology tree and open frontier.",
    )
    tree_parser.add_argument("--repo", default=".")
    tree_parser.add_argument("--json", action="store_true", dest="as_json")

    dag_parser = subparsers.add_parser(
        "dag",
        help="Build a unified evidence, verification, and agent-evolution DAG.",
    )
    dag_parser.add_argument("--repo", default=".")
    dag_parser.add_argument("--ref", default="HEAD")
    dag_parser.add_argument(
        "--ara",
        action="append",
        default=[],
        help=(
            "Additional ARA root whose experiment graph should be linked; "
            "committed ARAs bound at --ref are discovered automatically."
        ),
    )
    dag_parser.add_argument(
        "--output",
        help="Write research-dag.json and an offline research-dag.html browser.",
    )
    dag_parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Hide question, hypothesis, result, and claim summaries.",
    )
    dag_parser.add_argument("--json", action="store_true", dest="as_json")

    adapter_parser = subparsers.add_parser(
        "adapter",
        help="Discover and explicitly invoke versioned external platform adapters.",
    )
    adapter_subparsers = adapter_parser.add_subparsers(
        dest="adapter_command", required=True
    )
    adapter_list = adapter_subparsers.add_parser(
        "list",
        help="List built-in and installed third-party adapters without loading plugins.",
    )
    adapter_list.add_argument("--json", action="store_true", dest="as_json")
    adapter_doctor = adapter_subparsers.add_parser(
        "doctor", help="Load one selected adapter and check its requirements."
    )
    adapter_doctor.add_argument("name")
    adapter_doctor.add_argument("--json", action="store_true", dest="as_json")
    adapter_sync = adapter_subparsers.add_parser(
        "sync",
        help="Export a committed ref and publish it through one selected adapter.",
    )
    adapter_sync.add_argument("name")
    adapter_sync.add_argument("--repo", default=".")
    adapter_sync.add_argument("--ref", default="HEAD")
    adapter_sync.add_argument("--dest", required=True)
    adapter_sync.add_argument(
        "--format",
        action="append",
        choices=[
            "ro-crate",
            "prov-json",
            "cwl",
            "dvc",
            "mlflow",
            "openlineage",
            "croissant",
            "nanopub",
        ],
        default=[],
        dest="formats",
    )
    adapter_sync.add_argument("--include-payloads", action="store_true")
    adapter_sync.add_argument(
        "--option", action="append", default=[], help="Adapter option as NAME=VALUE."
    )
    adapter_sync.add_argument("--json", action="store_true", dest="as_json")

    fsck_parser = subparsers.add_parser(
        "fsck", help="Verify checkpoints, ARA bindings, pointers, and CAS objects."
    )
    fsck_parser.add_argument("--repo", default=".")
    fsck_parser.add_argument("--commit", default="HEAD")
    fsck_parser.add_argument("--no-objects", action="store_true")
    fsck_parser.add_argument("--json", action="store_true", dest="as_json")

    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="Commit a safe, reproducible scientific checkpoint."
    )
    checkpoint_parser.add_argument("--repo", default=".")
    checkpoint_parser.add_argument("--stage", required=True)
    checkpoint_parser.add_argument("--subject", required=True)
    checkpoint_parser.add_argument("--summary", default="")
    checkpoint_parser.add_argument("--status", default="completed")
    checkpoint_parser.add_argument("--actor")
    checkpoint_parser.add_argument("--node", action="append", default=[])
    checkpoint_parser.add_argument("--claim", action="append", default=[])
    checkpoint_parser.add_argument("--ara", action="append", default=[])
    checkpoint_parser.add_argument("--object-ref", action="append", default=[])
    checkpoint_parser.add_argument("--reproduce")
    checkpoint_parser.add_argument("--include", action="append", default=[])
    checkpoint_parser.add_argument("--no-commit", action="store_true")
    checkpoint_parser.add_argument("--allow-checkpoint-only", action="store_true")
    checkpoint_parser.add_argument(
        "--staged",
        action="store_true",
        help="Commit exactly the paths selected by `research stage`.",
    )
    checkpoint_parser.add_argument("--json", action="store_true", dest="as_json")

    record_parser = subparsers.add_parser(
        "record", help="Record one immutable, typed scientific object."
    )
    record_parser.add_argument("kind")
    record_parser.add_argument("--repo", default=".")
    record_parser.add_argument(
        "--state", choices=RESEARCH_OBJECT_STATES, default="draft"
    )
    record_parser.add_argument("--data", help="Payload as a JSON object.")
    record_parser.add_argument("--file", help="Read the JSON payload from a file.")
    record_parser.add_argument(
        "--profile-file",
        help=(
            "Semantic Profile descriptor JSON; required for extension kinds and "
            "content-bound into the object."
        ),
    )
    record_parser.add_argument(
        "--relation",
        action="append",
        default=[],
        help="Scientific relation as TYPE:TARGET[:ROLE].",
    )
    record_parser.add_argument("--json", action="store_true", dest="as_json")

    objects_parser = subparsers.add_parser(
        "objects", help="List or inspect typed scientific objects."
    )
    objects_parser.add_argument("object_id", nargs="?")
    objects_parser.add_argument("--repo", default=".")
    objects_parser.add_argument("--kind")
    objects_parser.add_argument("--state", choices=RESEARCH_OBJECT_STATES)
    objects_parser.add_argument("--json", action="store_true", dest="as_json")

    stage_parser = subparsers.add_parser(
        "stage", help="Select exact research changes for the next checkpoint."
    )
    stage_parser.add_argument("paths", nargs="*")
    stage_parser.add_argument("--repo", default=".")
    stage_parser.add_argument("--all", action="store_true", dest="all_changes")
    stage_parser.add_argument("--json", action="store_true", dest="as_json")

    add_parser = subparsers.add_parser(
        "add", help="Git-style alias for selecting exact research changes."
    )
    add_parser.add_argument("paths", nargs="*")
    add_parser.add_argument("--repo", default=".")
    add_parser.add_argument("-A", "--all", action="store_true", dest="all_changes")
    add_parser.add_argument("--json", action="store_true", dest="as_json")

    unstage_parser = subparsers.add_parser(
        "unstage", help="Remove paths from research staging without changing files."
    )
    unstage_parser.add_argument("paths", nargs="*")
    unstage_parser.add_argument("--repo", default=".")
    unstage_parser.add_argument("--all", action="store_true", dest="all_paths")
    unstage_parser.add_argument("--json", action="store_true", dest="as_json")

    commit_parser = subparsers.add_parser(
        "commit", help="Create a checkpoint from the native research stage."
    )
    commit_parser.add_argument("--repo", default=".")
    commit_parser.add_argument("-m", "--message", required=True)
    commit_parser.add_argument("--stage", default="research")
    commit_parser.add_argument("--summary", default="")
    commit_parser.add_argument("--status", default="completed")
    commit_parser.add_argument("--actor")
    commit_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all_changes",
        help="Select all eligible research changes before committing.",
    )
    commit_parser.add_argument("--json", action="store_true", dest="as_json")

    branch_parser = subparsers.add_parser(
        "branch", help="List or fork independent research lines."
    )
    branch_parser.add_argument("name", nargs="?")
    branch_parser.add_argument("--repo", default=".")
    branch_parser.add_argument("--from", default="HEAD", dest="from_ref")
    branch_parser.add_argument("--switch", action="store_true")
    branch_action = branch_parser.add_mutually_exclusive_group()
    branch_action.add_argument("-d", "--delete", action="store_true")
    branch_action.add_argument("-D", "--force-delete", action="store_true")
    branch_action.add_argument("-m", "--move", metavar="NEW_NAME")
    branch_parser.add_argument("--json", action="store_true", dest="as_json")

    switch_parser = subparsers.add_parser(
        "switch", help="Switch to another clean research line."
    )
    switch_parser.add_argument("name")
    switch_parser.add_argument("--repo", default=".")
    switch_parser.add_argument("--json", action="store_true", dest="as_json")

    restore_parser = subparsers.add_parser(
        "restore", help="Restore explicit research paths from a checkpoint."
    )
    restore_parser.add_argument("source")
    restore_parser.add_argument("paths", nargs="+")
    restore_parser.add_argument("--repo", default=".")
    restore_parser.add_argument("--json", action="store_true", dest="as_json")

    revert_parser = subparsers.add_parser(
        "revert", help="Revert a checkpoint and record the reversal scientifically."
    )
    revert_parser.add_argument("commit")
    revert_parser.add_argument("--repo", default=".")
    revert_parser.add_argument("-m", "--message")
    revert_parser.add_argument("--json", action="store_true", dest="as_json")

    tag_parser = subparsers.add_parser(
        "tag", help="List or immutably name scientific checkpoints."
    )
    tag_parser.add_argument("name", nargs="?")
    tag_parser.add_argument("--repo", default=".")
    tag_parser.add_argument("--commit", default="HEAD")
    tag_parser.add_argument("--annotation", default="")
    tag_parser.add_argument("--json", action="store_true", dest="as_json")

    blame_parser = subparsers.add_parser(
        "blame", help="Trace a scientific object to its originating checkpoint."
    )
    blame_parser.add_argument("object_id")
    blame_parser.add_argument("--repo", default=".")
    blame_parser.add_argument("--commit", default="HEAD")
    blame_parser.add_argument("--json", action="store_true", dest="as_json")

    merge_parser = subparsers.add_parser(
        "merge", help="Preflight or merge a scientifically compatible research line."
    )
    merge_parser.add_argument("source")
    merge_parser.add_argument("--repo", default=".")
    merge_parser.add_argument("--preview", action="store_true")
    merge_parser.add_argument("--subject")
    merge_parser.add_argument("--summary", default="")
    merge_parser.add_argument("--actor")
    merge_parser.add_argument(
        "--preserve-conflicts",
        action="store_true",
        help=(
            "preserve opposed evidence and add a rejected hold gate; other conflict "
            "types remain blocked"
        ),
    )
    merge_parser.add_argument("--json", action="store_true", dest="as_json")

    log_parser = subparsers.add_parser("log", help="Show scientific history.")
    log_parser.add_argument("--repo", default=".")
    log_parser.add_argument("--limit", type=int, default=20)
    log_parser.add_argument("--json", action="store_true", dest="as_json")

    trajectory_parser = subparsers.add_parser(
        "trajectory",
        help=(
            "Show the bounded, payload-free structured trajectory of typed "
            "objects and hash-valid checkpoints."
        ),
        description=(
            "Inspect the structured object-and-checkpoint history that makes "
            "Git-like scientific operations possible. The projection validates "
            "checkpoint hashes and parent edges, includes additions and legal "
            "revert removals, and never exposes object payloads or hidden reasoning."
        ),
    )
    trajectory_parser.add_argument(
        "--repo",
        default=".",
        help="research repository or a path inside it (default: current directory)",
    )
    trajectory_parser.add_argument(
        "--ref",
        default="HEAD",
        help="research branch, tag, or commit to project (default: HEAD)",
    )
    trajectory_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="maximum checkpoints to inspect, from 1 to 127 (default: 50)",
    )
    trajectory_parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit the complete machine-readable projection",
    )

    show_parser = subparsers.add_parser("show", help="Show a checkpoint at a commit.")
    show_parser.add_argument("commit", nargs="?", default="HEAD")
    show_parser.add_argument("--repo", default=".")
    show_parser.add_argument("--json", action="store_true", dest="as_json")

    diff_parser = subparsers.add_parser("diff", help="Compare two research commits.")
    diff_parser.add_argument("before", nargs="?", default="HEAD~1")
    diff_parser.add_argument("after", nargs="?", default="HEAD")
    diff_parser.add_argument("--repo", default=".")
    diff_parser.add_argument(
        "--deep",
        action="store_true",
        help="Compare structured scientific JSON fields up to the safety limit.",
    )
    diff_parser.add_argument("--json", action="store_true", dest="as_json")

    object_parser = subparsers.add_parser(
        "object", help="Register large evidence in the local content-addressed store."
    )
    object_subparsers = object_parser.add_subparsers(
        dest="object_command", required=True
    )
    object_add = object_subparsers.add_parser(
        "add", help="Add a file and write a Git-safe pointer."
    )
    object_add.add_argument("source")
    object_add.add_argument("--repo", default=".")
    object_add.add_argument("--logical-path")
    object_add.add_argument("--media-type")
    object_add.add_argument("--json", action="store_true", dest="as_json")

    bundle_parser = subparsers.add_parser(
        "bundle", help="Create, verify, or restore an offline research bundle."
    )
    bundle_parser.add_argument(
        "action",
        nargs="?",
        choices=["create", "verify", "restore"],
        default="create",
    )
    bundle_parser.add_argument("bundle_path", nargs="?")
    bundle_parser.add_argument("--repo", default=".")
    bundle_parser.add_argument("--dest")
    bundle_parser.add_argument(
        "--profile", choices=["index", "reproduce", "audit"], default="reproduce"
    )
    bundle_parser.add_argument("--allow-incomplete", action="store_true")
    bundle_parser.add_argument("--json", action="store_true", dest="as_json")

    export_parser = subparsers.add_parser(
        "export",
        help="Export one committed research state to standard ecosystem formats.",
    )
    export_parser.add_argument("--repo", default=".")
    export_parser.add_argument("--ref", default="HEAD")
    export_parser.add_argument("--dest", required=True)
    export_parser.add_argument(
        "--format",
        action="append",
        choices=["ro-crate", "prov-json", "cwl", "dvc", "mlflow"],
        default=[],
        dest="formats",
        help="Repeat to select formats; the default exports all adapters.",
    )
    export_parser.add_argument(
        "--include-payloads",
        action="store_true",
        help="Include scientific payloads in RO-Crate; metadata-only is safer by default.",
    )
    export_parser.add_argument("--json", action="store_true", dest="as_json")

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Inspect or materialize a commit's reproduction closure."
    )
    reproduce_parser.add_argument("commit", nargs="?", default="HEAD")
    reproduce_parser.add_argument("--repo", default=".")
    reproduce_parser.add_argument("--dest")
    reproduce_parser.add_argument("--execute", action="store_true")
    reproduce_parser.add_argument(
        "--record",
        action="store_true",
        help="Record the compact receipt back into Research VCS.",
    )
    reproduce_parser.add_argument(
        "--reproduces",
        action="append",
        default=[],
        help="Typed object ID checked by the receipt; repeat as needed.",
    )
    reproduce_parser.add_argument("--verifier")
    reproduce_parser.add_argument("--verified", action="store_true")
    reproduce_parser.add_argument("--no-commit", action="store_true")
    reproduce_parser.add_argument("--timeout", type=int, default=600)
    reproduce_parser.add_argument(
        "--environment-policy",
        choices=["ignore", "warn", "strict"],
        default="warn",
    )
    reproduce_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _human_status(payload: dict[str, Any]) -> None:
    print(f"Repository:        {_display_path(payload['repository'])}")
    print(f"Branch:            {payload['branch']}")
    print(f"HEAD:              {payload['head'] or '(no commit)'}")
    print(f"Checkpoint policy: {payload['checkpoint_policy']}")
    print(f"Auto push:         {payload['auto_push']}")
    print(f"Research staged:   {len(payload['research_stage']['paths'])}")
    print(f"Eligible changes:  {len(payload['eligible_changes'])}")
    print(f"Excluded changes:  {len(payload['excluded_changes'])}")
    for path in payload["eligible_changes"]:
        print(f"  eligible:        {path}")
    for path in payload["excluded_changes"]:
        print(f"  excluded:        {_display_text(path)}")
    store = payload["object_store"]
    print(f"Local CAS:         {store['objects']} objects / {store['bytes']} bytes")
    previous = payload.get("last_checkpoint") or {}
    if previous:
        print(
            f"Last checkpoint:   {previous.get('checkpoint_id')} "
            f"({previous.get('stage')} / {previous.get('status')})"
        )


def _object_summary(payload: dict[str, Any]) -> str:
    data = payload.get("payload") or {}
    for key in (
        "statement",
        "summary",
        "result",
        "title",
        "text",
        "decision",
        "status",
        "name",
    ):
        value = data.get(key)
        if value not in (None, "", [], {}):
            compact = " ".join(str(value).split())
            return compact[:77] + ("..." if len(compact) > 77 else "")
    return "(no summary)"


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "xscientist research",
) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .git_support import inspect_git_backend

            payload = inspect_git_backend()
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Research VCS backend: {payload['backend']}")
                print(f"Available:            {payload['available']}")
                print(f"Version:              {payload['version'] or 'N/A'}")
                for name, ready in payload["capabilities"].items():
                    print(f"{name.replace('_', ' ').title():<22} {ready}")
                for error in payload["errors"]:
                    print(f"error: {_display_text(error)}", file=sys.stderr)
                if payload.get("install_hint"):
                    print(_display_text(payload["install_hint"]), file=sys.stderr)
            return 0 if payload["ok"] else 1

        if args.command == "start":
            from .research_journey import (
                public_guided_research_start_payload,
                start_guided_research,
            )

            payload = start_guided_research(
                args.path,
                question=args.question,
                hypothesis=args.hypothesis,
                falsifier=args.falsifier,
                name=args.name,
                actor=args.actor,
                language=args.lang,
                git_user_name=args.git_user_name,
                git_user_email=args.git_user_email,
            )
            if args.as_json:
                _print_json(
                    public_guided_research_start_payload(
                        payload,
                        workspace=payload.get("repository") or args.path,
                    )
                )
            else:
                print(f"Research workspace: {_display_path(payload['repository'])}")
                print(f"Question:           {payload['question_id']}")
                print(f"Hypothesis:         {payload['hypothesis_id']}")
                print("Next step:")
                for step in payload["guide"]["next_steps"]:
                    print(f"  {step['title']}")
                    print(f"  Why: {_display_text(step['why'])}")
                    print(f"  Run: {step['command']}")
            return 0

        if args.command == "guide":
            from .research_journey import (
                build_research_guide,
                public_research_guide_payload,
            )

            payload = build_research_guide(args.repo, language=args.lang)
            if args.as_json:
                _print_json(public_research_guide_payload(payload))
            else:
                progress = payload["progress"]
                print(
                    f"Research progress: {progress['completed_stages']}/"
                    f"{progress['total_stages']} ({progress['percent']}%)"
                )
                for step in payload["next_steps"]:
                    print(f"\n{step['title']}")
                    print(f"  {_display_text(step['why'])}")
                    print(f"  {step['command']}")
                for warning in payload["warnings"]:
                    print(
                        f"warning: {_display_text(warning['message'])}",
                        file=sys.stderr,
                    )
            return 0

        if args.command == "hypothesis":
            from .research_commands import save_hypothesis

            result = save_hypothesis(
                args.repo,
                statement=args.statement,
                falsifier=args.falsifier,
                rationale=args.rationale,
                predictions=args.prediction,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "hypothesis", result, as_json=args.as_json, guide_repo=args.repo
            )
            return 0

        if args.command == "plan":
            from .research_commands import save_research_plan

            result = save_research_plan(
                args.repo,
                hypothesis_id=args.hypothesis_id,
                summary=args.summary,
                discriminating_tests=args.test,
                success_rule=args.success_rule,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "research plan", result, as_json=args.as_json, guide_repo=args.repo
            )
            return 0

        if args.command == "discovery":
            from .research_discovery import (
                discovery_contract_template,
                save_discovery_contract,
                save_generalization_assessment,
            )

            if args.discovery_command == "template":
                payload = discovery_contract_template()
                if args.output:
                    from ai_scientist.utils.atomic_io import atomic_write_json

                    destination = Path(args.output).expanduser()
                    if destination.exists():
                        raise ResearchGitError(
                            "discovery template output already exists; choose a new path"
                        )
                    atomic_write_json(destination, payload, ensure_ascii=False)
                    if args.as_json:
                        _print_json({"output": str(destination), "template": payload})
                    else:
                        print(
                            "Discovery contract template: "
                            f"{_display_path(destination)}"
                        )
                        print("Replace every REPLACE_* value before locking the plan.")
                else:
                    _print_json(payload)
                return 0
            if args.discovery_command == "plan":
                result = save_discovery_contract(
                    args.repo,
                    hypothesis_id=args.hypothesis_id,
                    spec=_read_json_mapping(
                        args.spec, label="method discovery specification"
                    ),
                    context_id=args.context,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "method discovery contract", result, as_json=args.as_json
                )
                return 0
            result = save_generalization_assessment(
                args.repo,
                contract_id=args.contract_id,
                results=_read_json_mapping(
                    args.results, label="method discovery results"
                ),
                evidence_ids=args.evidence,
                message=args.message,
                commit=not args.no_commit,
            )
            if args.as_json:
                recorded = result["object"]
                checkpoint = result.get("checkpoint")
                _print_json(
                    {
                        "object": recorded.to_dict(),
                        "assessment": result["assessment"],
                        "checkpoint": (
                            checkpoint.to_dict() if checkpoint is not None else None
                        ),
                    }
                )
            else:
                _print_saved_object("generalization assessment", result, as_json=False)
                assessment = result["assessment"]
                print(f"Verdict:    {assessment['verdict']}")
                passed = sum(
                    row.get("passed") is True
                    for row in assessment["condition_assessments"]
                )
                print(
                    "Conditions: "
                    f"{passed}/{len(assessment['condition_assessments'])} passed"
                )
            return 0

        if args.command == "program":
            from .research_strategy import (
                inspect_claim_depth,
                rank_experiment_candidates,
                record_research_followup_queue,
                research_strategy_template,
                review_research_program,
                save_discriminating_prediction,
                save_evidence_quality_assessment,
                save_hypothesis_portfolio,
                save_mechanism_model,
                save_posterior_update,
                save_transfer_matrix,
            )

            if args.program_command == "template":
                payload = research_strategy_template()
                if args.output:
                    from ai_scientist.utils.atomic_io import atomic_write_json

                    destination = Path(args.output).expanduser()
                    if destination.exists():
                        raise ResearchGitError(
                            "research strategy template output already exists"
                        )
                    atomic_write_json(destination, payload, ensure_ascii=False)
                    response = {"output": str(destination), "template": payload}
                    if args.as_json:
                        _print_json(response)
                    else:
                        print(
                            "Deep-research template: " f"{_display_path(destination)}"
                        )
                else:
                    _print_json(payload)
                return 0
            if args.program_command == "portfolio":
                raw_priors = _parse_assignments(args.prior, label="prior")
                try:
                    priors = {key: float(value) for key, value in raw_priors.items()}
                except (TypeError, ValueError) as exc:
                    raise ResearchGitError(
                        "prior weights must be numeric NAME=VALUE assignments"
                    ) from exc
                result = save_hypothesis_portfolio(
                    args.repo,
                    question=args.question,
                    primary_id=args.primary_id,
                    alternative_ids=args.alternative,
                    null_id=args.null_id,
                    prior_weights=priors,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "hypothesis portfolio", result, as_json=args.as_json
                )
                return 0
            if args.program_command == "prediction":
                result = save_discriminating_prediction(
                    args.repo,
                    portfolio_id=args.portfolio_id,
                    hypothesis_id=args.hypothesis_id,
                    when=args.when,
                    expected_outcome=args.expect,
                    distinguishes_from=args.distinguishes,
                    falsifier=args.falsifier,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "discriminating prediction", result, as_json=args.as_json
                )
                return 0
            if args.program_command == "prioritize":
                raw_candidates = _read_json_value(
                    args.candidates, label="experiment candidates"
                )
                if isinstance(raw_candidates, dict):
                    raw_candidates = raw_candidates.get("experiment_candidates")
                if not isinstance(raw_candidates, list):
                    raise ResearchGitError(
                        "experiment candidates must be a JSON array or contain experiment_candidates"
                    )
                result = rank_experiment_candidates(
                    args.repo,
                    portfolio_id=args.portfolio_id,
                    candidates=raw_candidates,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("experiment priority", result, as_json=args.as_json)
                if not args.as_json:
                    selected = result["ranking"]["candidate_set"][0]
                    print(
                        f"Next experiment: {selected['candidate_id']} "
                        f"(EIG={selected['expected_information_gain']}, "
                        f"utility={selected['utility_score']})"
                    )
                return 0
            if args.program_command == "posterior":
                raw_likelihoods = _parse_assignments(
                    args.likelihood, label="likelihood"
                )
                try:
                    likelihoods = {
                        key: float(value) for key, value in raw_likelihoods.items()
                    }
                except (TypeError, ValueError) as exc:
                    raise ResearchGitError(
                        "likelihoods must be numeric hypothesis=probability assignments"
                    ) from exc
                result = save_posterior_update(
                    args.repo,
                    portfolio_id=args.portfolio_id,
                    priority_id=args.priority_id,
                    attempt_id=args.attempt_id,
                    evidence_id=args.evidence_id,
                    observed_outcome=args.observed,
                    likelihoods=likelihoods,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("posterior update", result, as_json=args.as_json)
                return 0
            if args.program_command == "mechanism":
                result = save_mechanism_model(
                    args.repo,
                    hypothesis_id=args.hypothesis_id,
                    statement=args.statement,
                    mediators=args.mediator,
                    interventions=args.intervention,
                    rival_hypothesis_ids=args.rival,
                    evidence_ids=args.evidence,
                    status=args.status,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("mechanism model", result, as_json=args.as_json)
                return 0
            if args.program_command == "quality":
                assessment = _read_json_mapping(
                    args.assessment, label="evidence quality assessment"
                )
                domains = assessment.get("domains", assessment)
                if not isinstance(domains, dict):
                    raise ResearchGitError(
                        "quality assessment domains must be an object"
                    )
                notes = assessment.get("notes") or {}
                if not isinstance(notes, dict):
                    raise ResearchGitError("quality assessment notes must be an object")
                result = save_evidence_quality_assessment(
                    args.repo,
                    evidence_id=args.evidence_id,
                    domains=domains,
                    notes=notes,
                    independent=args.independent,
                    assessor_id=args.assessor,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("evidence quality", result, as_json=args.as_json)
                return 0
            if args.program_command == "boundary":
                raw_rows = _read_json_value(args.matrix, label="boundary matrix")
                if isinstance(raw_rows, dict):
                    raw_rows = raw_rows.get("boundary_rows")
                if not isinstance(raw_rows, list):
                    raise ResearchGitError(
                        "boundary matrix must be a JSON array or contain boundary_rows"
                    )
                result = save_transfer_matrix(
                    args.repo,
                    claim_id=args.claim_id,
                    rows=raw_rows,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("transfer matrix", result, as_json=args.as_json)
                return 0
            if args.program_command == "claim":
                payload = inspect_claim_depth(args.repo, args.claim_id)
                if args.as_json:
                    _print_json(payload)
                else:
                    print(f"Claim:         {payload['claim_id']}")
                    print(f"Depth:         {payload['depth_level']}")
                    print(
                        f"Decision ready:{' yes' if payload['decision_ready'] else ' no'}"
                    )
                    print(f"Support:       {len(payload['supporting_ids'])}")
                    print(f"Refutation:    {len(payload['refuting_ids'])}")
                    print(f"Mechanisms:    {len(payload['mechanism_ids'])}")
                    print(f"Quality audits:{len(payload['quality_assessment_ids'])}")
                    print(f"Boundaries:    {len(payload['boundary_ids'])}")
                    for gap in payload["gaps"]:
                        print(f"  gap: {gap}")
                    if payload["next_experiment"]:
                        print(
                            "Next experiment: "
                            f"{payload['next_experiment']['summary']}"
                        )
                return 0
            if args.program_command == "followup":
                payload = record_research_followup_queue(
                    args.repo,
                    review_id=args.review,
                    max_actions=args.max_actions,
                    message=args.message,
                    commit=not args.no_commit,
                )
                if args.as_json:
                    checkpoint = payload.get("checkpoint")
                    _print_json(
                        {
                            **payload,
                            "checkpoint": (
                                checkpoint.to_dict() if checkpoint is not None else None
                            ),
                        }
                    )
                else:
                    print(
                        "Scientific strategy follow-ups: "
                        f"created {len(payload['queued'])}; "
                        f"active {len(payload['active'])}/{payload['max_actions']}"
                    )
                    for item in payload["active"]:
                        print(f"  {item['object_id']}: {_display_text(item['action'])}")
                return 0
            result = review_research_program(
                args.repo,
                record=args.record,
                message=args.message,
                commit=not args.no_commit,
            )
            if args.as_json:
                _print_json(
                    {
                        "report": result["report"],
                        "object": (
                            result["object"].to_dict()
                            if result["object"] is not None
                            else None
                        ),
                        "checkpoint": (
                            result["checkpoint"].to_dict()
                            if result["checkpoint"] is not None
                            else None
                        ),
                    }
                )
            else:
                report = result["report"]
                print(report["summary"])
                print(f"Review due: {report['review_due']}")
                for gap in report["gaps"]:
                    print(f"  {gap['code']}: {_display_text(gap['message'])}")
                if result["object"] is not None:
                    print(f"Recorded review: {result['object'].object_id}")
            return 0

        if args.command == "opportunity":
            if args.opportunity_command == "direction":
                result = save_research_direction(
                    args.repo,
                    direction_id=args.direction_id,
                    statement=args.statement,
                    objective=args.objective,
                    domain=args.domain,
                    success_definition=args.success_definition,
                    constraints=args.constraint,
                    source_refs=args.source_ref,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("research direction", result, as_json=args.as_json)
                return 0
            if args.opportunity_command == "pool":
                raw_candidates = _read_json_value(
                    args.candidates, label="opportunity candidates"
                )
                if isinstance(raw_candidates, dict):
                    raw_candidates = raw_candidates.get("candidates")
                if not isinstance(raw_candidates, list):
                    raise ResearchGitError(
                        "opportunity candidates must be a JSON array or contain candidates"
                    )
                result = save_opportunity_pool(
                    args.repo,
                    direction_id=args.direction_id,
                    candidates=raw_candidates,
                    complete_candidate_set=not args.incomplete,
                    extraction_notes=args.extraction_notes,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("opportunity pool", result, as_json=args.as_json)
                return 0
            if args.opportunity_command == "attempt":
                result = save_opportunity_attempt(
                    args.repo,
                    pool_id=args.pool_id,
                    candidate_id=args.candidate_id,
                    outcome=args.outcome,
                    summary=args.summary,
                    evidence_refs=args.evidence_ref,
                    evidence_object_ids=args.evidence_object_id,
                    runner=args.runner,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("opportunity attempt", result, as_json=args.as_json)
                return 0
            if args.opportunity_command == "judge":
                result = save_opportunity_judgment(
                    args.repo,
                    attempt_id=args.attempt_id,
                    verdict=args.verdict,
                    evaluator_id=args.evaluator_id,
                    summary=args.summary,
                    evidence_refs=args.evidence_ref,
                    evidence_object_ids=args.evidence_object_id,
                    allow_stage_override=args.allow_stage_override,
                    override_reason=args.override_reason,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "opportunity judgment", result, as_json=args.as_json
                )
                return 0
            if args.opportunity_command == "grade":
                result = save_opportunity_grade(
                    args.repo,
                    judgment_id=args.judgment_id,
                    grade=args.grade,
                    evaluator_id=args.evaluator_id,
                    summary=args.summary,
                    evidence_refs=args.evidence_ref,
                    evidence_object_ids=args.evidence_object_id,
                    allow_stage_override=args.allow_stage_override,
                    override_reason=args.override_reason,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("opportunity grade", result, as_json=args.as_json)
                return 0
            if args.opportunity_command == "allocate":
                result = save_opportunity_allocation(
                    args.repo,
                    pool_id=args.pool_id,
                    objective=args.objective,
                    max_attempts=args.max_attempts,
                    calibration_status=args.calibration_status,
                    probability_semantics=args.probability_semantics,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "opportunity allocation", result, as_json=args.as_json
                )
                return 0
            inspected = inspect_opportunity_funnel(args.repo, args.pool_id)
            if args.as_json:
                _print_json(inspected)
            else:
                summary = inspected["summary"]
                print(
                    f"Opportunity funnel: {summary['attempted_candidate_count']}/"
                    f"{summary['candidate_count']} candidates attempted"
                )
                print(f"Outcomes: {summary['outcome_counts']}")
                print(f"Unattempted: {summary['unattempted_candidate_ids']}")
                print(f"Funnel complete: {summary['funnel_complete']}")
                return 0

        if args.command == "rollout":
            episode = _read_json_mapping(args.episode, label="research rollout")
            result = save_research_rollout(
                args.repo,
                episode,
                message=args.message,
                commit=not args.no_commit,
            )
            if args.as_json:
                _print_json(
                    {
                        **_saved_object_json(result),
                        "rollout": result["rollout"],
                    }
                )
            else:
                _print_saved_object("research rollout", result, as_json=False)
            return 0

        if args.command == "rollout-audit":
            raw_report = _read_json_mapping(args.report, label="rollout audit report")
            # Accept both the raw payload and the redacted JSON wrapper emitted
            # by `research rollout --json`.
            report = raw_report
            if isinstance(raw_report.get("rollout"), dict):
                report = raw_report["rollout"]
            object_payload = raw_report.get("object")
            if report is raw_report and isinstance(object_payload, dict):
                candidate = object_payload.get("payload")
                if isinstance(candidate, dict):
                    report = candidate
            trust_store = (
                _read_json_mapping(args.trust_store, label="attestation trust store")
                if args.trust_store
                else None
            )
            audit = audit_research_rollout(
                report,
                evidence_hashes=args.evidence_hash or None,
                trust_store=trust_store,
                max_attestation_age_seconds=args.max_attestation_age_seconds,
            )
            if args.as_json:
                _print_json(audit)
            else:
                print(f"Rollout audit: {audit['status']}")
                print(f"Verification allowed: {audit['verification_allowed']}")
                print(f"Blockers: {len(audit['blockers'])}")
                for item in audit["blockers"]:
                    print(f"  {item['code']}: {_display_text(item['message'])}")
                if audit["warnings"]:
                    print(f"Warnings: {len(audit['warnings'])}")
            return 0 if audit["verification_allowed"] else 1

        if args.command == "verifier-authority":
            from ai_scientist.protocol.canonical_json import canonical_content_hash
            from ai_scientist.utils.verifier_authority import (
                VERIFIER_AUTHORITY_PURPOSE,
            )

            from .publication_authority import (
                PublicationAuthorityError,
                finalize_verifier_authority_receipt,
                prepare_verifier_authority_payload,
                verify_publication_authority,
                write_publication_authority_json,
            )

            try:
                if args.verifier_authority_command == "prepare":
                    payload = prepare_verifier_authority_payload(
                        args.paper_dir,
                        verifier_identity=args.identity,
                    )
                    output_path = write_publication_authority_json(
                        args.output,
                        payload,
                        force=args.force,
                    )
                    result = {
                        "schema_version": "xscientist.verifier-authority-cli.v1",
                        "status": "prepared",
                        "purpose": VERIFIER_AUTHORITY_PURPOSE,
                        "output": _display_path(output_path),
                        "payload_hash": canonical_content_hash(payload),
                        "next_command": shlex.join(
                            [
                                "xscientist",
                                "evolution",
                                "attest",
                                "sign",
                                "--payload",
                                _display_path(output_path),
                                "--purpose",
                                VERIFIER_AUTHORITY_PURPOSE,
                                "--identity",
                                str(payload.get("verifier_identity") or args.identity),
                                "--key-id",
                                "KEY_ID",
                                "--private-key",
                                "VERIFIER_PRIVATE_KEY",
                                "--out",
                                "attestation.json",
                            ]
                        ),
                    }
                    if args.as_json:
                        _print_json(result)
                    else:
                        print(f"Authority payload: {_display_path(output_path)}")
                        print(f"Next: {result['next_command']}")
                    return 0

                if args.verifier_authority_command == "finalize":
                    attestation = _read_json_mapping(
                        args.attestation,
                        label="external verifier attestation",
                    )
                    receipt = finalize_verifier_authority_receipt(
                        args.paper_dir,
                        verifier_identity=args.identity,
                        attestation=attestation,
                    )
                    output_value = args.output or str(
                        Path(args.paper_dir).expanduser()
                        / "verifier_authority_receipt.json"
                    )
                    output_path = write_publication_authority_json(
                        output_value,
                        receipt,
                        force=args.force,
                    )
                    result = {
                        "schema_version": "xscientist.verifier-authority-cli.v1",
                        "status": "finalized",
                        "output": _display_path(output_path),
                        "receipt_hash": receipt.get("receipt_hash"),
                    }
                    if args.as_json:
                        _print_json(result)
                    else:
                        print(f"Authority receipt: {_display_path(output_path)}")
                        print(f"Receipt hash:      {receipt.get('receipt_hash')}")
                    return 0

                result = verify_publication_authority(
                    args.paper_dir,
                    trust_store=args.trust_store,
                    receipt_path=args.receipt,
                )
                if args.as_json:
                    _print_json(result)
                else:
                    print(f"Signature binding: {result['status']}")
                    print(
                        "Submission ready:  unknown "
                        "(run the complete scientific evidence gate)"
                        if result.get("signature_binding_verified") is True
                        else "Submission ready:  false"
                    )
                    for error in result.get("errors") or []:
                        print(f"  {_display_text(error)}")
                return 0 if result.get("ok") is True else 1
            except PublicationAuthorityError as exc:
                raise ResearchGitError(str(exc)) from exc

        if args.command == "belief":
            from .research_context import build_research_context_snapshot

            context = build_research_context_snapshot(
                args.repo,
                target_ids=args.target,
                intent="audit",
                decision_kind="belief_context_review",
                ref=args.ref,
                budget_tokens=args.budget,
                belief_as_of=args.as_of,
            )
            projection = context["belief_context"]
            if args.as_json:
                _print_json(projection)
            else:
                print(f"Belief context: {projection['projection_hash']}")
                print(f"As of:          {projection['as_of'] or 'unavailable'}")
                print(f"Complete:       {projection['complete']}")
                print(f"Conflicts:      {len(projection['conflict_sets'])}")
                for item in projection["target_assessments"]:
                    print(
                        f"  {item['target_id']}: {item['belief_state']} -> "
                        f"{item['decision_posture']}"
                    )
            return 0 if projection["complete"] else 1

        if args.command == "belief-audit":
            raw_report = _read_json_mapping(args.report, label="belief audit report")
            report = raw_report
            if isinstance(raw_report.get("context"), dict):
                report = raw_report["context"]
            if isinstance(report.get("belief_context"), dict):
                report = report["belief_context"]
            audit = audit_belief_context_projection(report)
            if args.as_json:
                _print_json(audit)
            else:
                status = "passed" if audit["verification_allowed"] else "blocked"
                print(f"Belief audit: {status}")
                print(f"Projection:   {audit['projection_hash'] or 'unavailable'}")
                print(f"Issues:       {len(audit['issues'])}")
                for issue in audit["issues"]:
                    print(f"  {issue}")
            return 0 if audit["verification_allowed"] else 1

        if args.command == "literature":
            from .research_commands import (
                save_passage_evidence,
                save_search_plan,
                save_search_receipt,
                save_source_snapshot,
                save_source_update,
            )

            if args.literature_command == "plan":
                result = save_search_plan(
                    args.repo,
                    question=args.question,
                    queries=args.query,
                    providers=args.provider,
                    inclusion_criteria=args.include,
                    exclusion_criteria=args.exclude,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "literature search plan", result, as_json=args.as_json
                )
                return 0
            if args.literature_command == "receipt":
                path = Path(args.results).expanduser()
                if not path.is_file():
                    raise ResearchGitError("literature results file was not found")
                try:
                    raw_results = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ResearchGitError(
                        "literature results file is invalid JSON"
                    ) from exc
                candidates = (
                    raw_results.get("candidates")
                    if isinstance(raw_results, dict)
                    else raw_results
                )
                if not isinstance(candidates, list) or not all(
                    isinstance(item, dict) for item in candidates
                ):
                    raise ResearchGitError(
                        "literature results must be an array of candidate objects"
                    )
                result = save_search_receipt(
                    args.repo,
                    plan_id=args.plan_id,
                    provider=args.provider,
                    query=args.query,
                    candidates=candidates,
                    retrieved_at=args.retrieved_at,
                    corpus_version=args.corpus_version,
                    corpus_snapshot_hash=args.corpus_snapshot_hash,
                    query_rewrites=args.query_rewrite,
                    filters=_parse_assignments(args.filter, label="literature filter"),
                    retrieval_system={
                        key: value
                        for key, value in {
                            "retriever": args.retriever,
                            "embedding_model": args.embedding_model,
                            "reranker": args.reranker,
                        }.items()
                        if value
                    },
                    pagination={
                        key: value
                        for key, value in {
                            "cursor": args.cursor,
                            "page": args.page,
                        }.items()
                        if value not in (None, "")
                    },
                    complete=not args.incomplete,
                    errors=args.error,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "literature search receipt", result, as_json=args.as_json
                )
                return 0
            if args.literature_command == "source":
                result = save_source_snapshot(
                    args.repo,
                    receipt_id=args.receipt_id,
                    title=args.title,
                    content_hash=args.content_hash or _hash_local_file(args.file),
                    metadata_hash=args.metadata_hash,
                    doi=args.doi,
                    pmid=args.pmid,
                    arxiv_id=args.arxiv_id,
                    url=args.url,
                    license_name=args.license_name,
                    retraction_status=args.retraction_status,
                    status_provider=args.status_provider,
                    status_checked_at=args.status_checked_at,
                    status_notice_id=args.status_notice_id,
                    previous_source_id=args.previous_source_id,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object("literature source", result, as_json=args.as_json)
                return 0
            if args.literature_command == "update":
                result = save_source_update(
                    args.repo,
                    source_id=args.source_id,
                    status=args.status,
                    provider=args.provider,
                    checked_at=args.checked_at,
                    update_type=args.update_type,
                    notice_id=args.notice_id,
                    detail=args.detail,
                    message=args.message,
                    commit=not args.no_commit,
                )
                _print_saved_object(
                    "literature source update", result, as_json=args.as_json
                )
                return 0
            result = save_passage_evidence(
                args.repo,
                source_id=args.source_id,
                quote=args.quote,
                locator=args.locator,
                prefix=args.prefix,
                suffix=args.suffix,
                start=args.start,
                end=args.end,
                supports=args.supports,
                refutes=args.refutes,
                context_id=args.context,
                scope=args.scope,
                structured_scope=_scope_from_args(args),
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("passage evidence", result, as_json=args.as_json)
            return 0

        if args.command == "confirm":
            from .research_commands import confirm_paper_research

            result = confirm_paper_research(
                args.paper_dir,
                registered_by=args.registered_by,
                split_hashes=_parse_confirmatory_splits(
                    args.split,
                    args.split_file,
                ),
                data_manifest_hash=args.data_manifest_hash,
                data_snapshot_id=args.data_snapshot_id,
                message=args.message,
            )
            if args.as_json:
                payload = _saved_object_json(result)
                payload.update(
                    {
                        "hypothesis_checkpoint": (
                            result["hypothesis_checkpoint"].to_dict()
                            if result.get("hypothesis_checkpoint") is not None
                            else None
                        ),
                        "campaign_checkpoint": result["campaign_checkpoint"].to_dict(),
                        "preregistration_path": result["preregistration_path"],
                        "queue_path": result["queue_path"],
                        "queue": result["queue"],
                        "validation": result["validation"],
                    }
                )
                _print_json(payload)
            else:
                print("Locked preregistration: " f"{result['object'].object_id}")
                print(f"Plan object:            {result['related'][0].object_id}")
                print(
                    "Freeze checkpoint:      " f"{result['checkpoint'].checkpoint_id}"
                )
                print(
                    "Queue checkpoint:       "
                    f"{result['campaign_checkpoint'].checkpoint_id}"
                )
                print(f"Queued tasks:           {len(result['queue']['tasks'])}")
                print(f"Queue:                  {_display_path(result['queue_path'])}")
            return 0

        if args.command == "trajectory-bind":
            from .research_commands import bind_experiment_trajectory

            result = bind_experiment_trajectory(
                args.paper_dir,
                record_id=args.record_id,
                attempt_id=args.attempt,
                message=args.message,
            )
            _print_saved_object("trajectory binding", result, as_json=args.as_json)
            return 0

        if args.command == "attempt-disposition":
            from .research_commands import record_attempt_disposition

            result = record_attempt_disposition(
                args.paper_dir,
                record_id=args.record_id,
                disposition=args.disposition,
                reason=args.reason,
                retry_record_id=args.retry_record_id,
                approved_before_unblinding=args.approved_before_unblinding,
                negative_result_artifact=args.negative_result_artifact,
                negative_result_evidence_id=args.negative_result_evidence,
                recorded_by=args.recorded_by,
                message=args.message,
            )
            _print_saved_object("attempt disposition", result, as_json=args.as_json)
            return 0

        if args.command == "experiment":
            from .research_commands import save_experiment

            result_artifact_paths = _parse_path_assignments(
                args.result_artifact, label="result artifact"
            )

            result = save_experiment(
                args.repo,
                summary=args.summary,
                status=args.status,
                study_phase=args.study_phase,
                task_id=args.task,
                plan_id=args.plan,
                priority_id=args.priority,
                preregistration_id=args.preregistration,
                metrics=_parse_assignments(args.metric, label="metric"),
                configuration=_parse_assignments(args.config, label="configuration"),
                producer_id=args.producer_id,
                result_artifact_paths=result_artifact_paths,
                seeds=args.seed,
                environment_hash=args.environment_hash,
                dependency_lock_hashes=[
                    *args.dependency_lock_hash,
                    *(_hash_local_file(path) for path in args.dependency_lock_file),
                ],
                dataset_hashes=args.dataset_hash,
                code_commit=args.code_commit,
                failure_class=args.failure_class,
                interventions=args.intervention,
                boundary_condition=args.boundary_condition,
                boundary_role=args.boundary_role,
                reproduce_command=args.reproduce_command,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "experiment", result, as_json=args.as_json, guide_repo=args.repo
            )
            return 0

        if args.command == "preregister":
            from .research_commands import save_preregistration

            result = save_preregistration(
                args.repo,
                hypothesis_id=args.hypothesis_id,
                dataset=args.dataset,
                metric=args.metric,
                baseline=args.baseline,
                split_hash=args.split_hash or _hash_local_file(args.split_file),
                registered_by=args.registered_by,
                minimum_effect=args.minimum_effect,
                alpha=args.alpha,
                minimum_seeds=args.minimum_seeds,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("preregistration", result, as_json=args.as_json)
            return 0

        if args.command == "evidence":
            from .research_commands import save_evidence

            result = save_evidence(
                args.repo,
                result_summary=args.result,
                attempt_ids=args.attempt,
                supports=args.supports,
                refutes=args.refutes,
                metrics=_parse_assignments(args.metric, label="metric"),
                scope=args.scope,
                structured_scope=_scope_from_args(args),
                verified=args.verified,
                verifier_id=args.verifier,
                reproduce_command=args.reproduce_command,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "evidence", result, as_json=args.as_json, guide_repo=args.repo
            )
            return 0

        if args.command == "estimand":
            from .research_commands import save_estimand

            result = save_estimand(
                args.repo,
                outcome=args.outcome,
                population=args.population,
                intervention=args.intervention,
                comparator=args.comparator,
                time_window=args.time_window,
                summary_measure=args.summary_measure,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("estimand", result, as_json=args.as_json)
            return 0

        if args.command == "effect":
            from .research_commands import save_effect_estimate

            result = save_effect_estimate(
                args.repo,
                estimand_id=args.estimand_id,
                estimate=args.estimate,
                metric=args.metric,
                unit=args.unit,
                confidence_level=args.confidence_level,
                interval_lower=args.lower,
                interval_upper=args.upper,
                standard_error=args.standard_error,
                derived_from=args.derived_from,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("effect estimate", result, as_json=args.as_json)
            return 0

        if args.command == "infer":
            from .research_commands import save_inference

            result = save_inference(
                args.repo,
                statement=args.statement,
                premises=args.premise,
                warrant=args.warrant,
                method_ids=args.method,
                assumption_ids=args.assumption,
                context_id=args.context,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "scientific inference",
                result,
                as_json=args.as_json,
                guide_repo=args.repo,
            )
            return 0

        if args.command == "ingest":
            from .research_tools import ingest_tool_evidence, load_tool_evidence

            result = ingest_tool_evidence(
                args.repo,
                load_tool_evidence(args.file),
                attempt_ids=args.attempt,
                supports=args.supports,
                refutes=args.refutes,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("tool evidence", result, as_json=args.as_json)
            return 0

        if args.command == "claim":
            from .research_commands import save_claim

            result = save_claim(
                args.repo,
                statement=args.statement,
                evidence_ids=args.evidence,
                scope=args.scope,
                structured_scope=_scope_from_args(args),
                contribution_level=args.contribution_level,
                depth_level=args.depth_level,
                mechanism_ids=args.mechanism,
                quality_ids=args.quality,
                transfer_ids=args.transfer,
                gate_id=args.gate,
                verified=args.verified,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "claim", result, as_json=args.as_json, guide_repo=args.repo
            )
            return 0

        if args.command == "review":
            from .research_commands import save_review

            result = save_review(
                args.repo,
                summary=args.summary,
                evaluates=args.evaluates,
                verifier_id=args.verifier,
                decision=args.decision,
                required_failures=args.failure,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object(
                "gate decision", result, as_json=args.as_json, guide_repo=args.repo
            )
            return 0

        if args.command == "init":
            result = init_repository(
                args.path,
                name=args.name,
                question=_read_question(args),
                policy=args.policy,
                actor=args.actor,
                git_user_name=args.git_user_name,
                git_user_email=args.git_user_email,
                max_file_bytes=args.max_file_bytes,
                commit=not args.no_commit,
            )
            payload = result.to_dict()
            if args.as_json:
                _print_json(payload)
            else:
                print(
                    "Initialized local research repository: "
                    f"{_display_path(args.path)}"
                )
                if result.commit:
                    print(
                        f"Initial checkpoint: {result.commit[:12]} {result.checkpoint_id}"
                    )
                print("Remote: none (XScientist never pushes automatically)")
            return 0

        if args.command == "status":
            payload = repository_status(args.repo)
            _print_json(payload) if args.as_json else _human_status(payload)
            return 0

        if args.command == "audit":
            from .research_closure import (
                audit_research_closure,
                summarize_closure_levels,
            )

            payload = audit_research_closure(
                args.repo,
                ref=args.ref,
                level=args.level,
                verify_objects=not args.no_objects,
            )
            if args.as_json:
                _print_json(payload)
            else:
                levels = summarize_closure_levels(payload)
                target_label = {
                    "trace": "Traceability closure",
                    "replay": "Replayability closure",
                    "verify": "Verification closure",
                }[payload["target_level"]]
                print(f"{target_label}: {payload['status']}")
                print(f"Target level:       {payload['target_level']}")
                print(
                    "Closure levels:      "
                    f"trace={'complete' if levels['trace'] else 'blocked'}, "
                    f"replay={'complete' if levels['replay'] else 'blocked'}, "
                    "verification="
                    f"{'complete' if levels['verify'] else 'blocked'}"
                )
                print(
                    "Overall scientific closure: "
                    f"{'complete' if levels['verify'] else 'pending'}"
                )
                print(f"Commit:             {payload['commit']}")
                print(f"Claims:             {len(payload['claims'])}")
                print(f"Blockers:           {len(payload['blockers'])}")
                for item in payload["blockers"]:
                    target = f" ({item['object_id']})" if item["object_id"] else ""
                    print(f"  {item['code']}{target}: {_display_text(item['message'])}")
                for item in payload["warnings"]:
                    target = f" ({item['object_id']})" if item["object_id"] else ""
                    print(
                        f"warning: {item['code']}{target}: "
                        f"{_display_text(item['message'])}",
                        file=sys.stderr,
                    )
            return 0 if payload["complete"] else 1

        if args.command == "context":
            from .research_context import (
                build_research_context_snapshot,
                record_research_context_snapshot,
                render_research_context_for_prompt,
            )

            options = _parse_context_options(args.option, selected=args.selected)
            if args.record:
                if args.ref not in {"WORKTREE", "worktree"}:
                    raise ResearchGitError(
                        "recorded context must use the current worktree; omit --ref"
                    )
                recorded = record_research_context_snapshot(
                    args.repo,
                    target_ids=args.target,
                    intent=args.intent,
                    decision_kind=args.decision_kind,
                    selected=args.selected,
                    options_considered=options,
                    rationale=args.rationale,
                    constraints=args.constraint,
                    memory_refs=args.memory_ref,
                    budget_tokens=args.budget,
                    belief_as_of=args.belief_as_of,
                )
                context_payload = load_research_object(args.repo, recorded.object_id)[
                    "payload"
                ]
                checkpoint = (
                    create_checkpoint(
                        args.repo,
                        stage="context",
                        subject=f"record {args.decision_kind} context",
                        include=[
                            recorded.path.relative_to(
                                recorded.path.parents[3]
                            ).as_posix(),
                            *(
                                f".xscientist/objects/{source['kind']}/"
                                f"{source['object_id']}.json"
                                for source in context_payload["source_objects"]
                            ),
                        ],
                    )
                    if not args.no_commit
                    else None
                )
                payload = {
                    "object": recorded.to_dict(),
                    "context": context_payload,
                    "checkpoint": checkpoint.to_dict() if checkpoint else None,
                }
            else:
                payload = build_research_context_snapshot(
                    args.repo,
                    target_ids=args.target,
                    intent=args.intent,
                    decision_kind=args.decision_kind,
                    selected=args.selected,
                    options_considered=options,
                    rationale=args.rationale,
                    constraints=args.constraint,
                    memory_refs=args.memory_ref,
                    ref=args.ref,
                    budget_tokens=args.budget,
                    belief_as_of=args.belief_as_of,
                )
            context = payload.get("context") or payload
            if args.prompt:
                print(render_research_context_for_prompt(context))
            elif args.as_json:
                _print_json(payload)
            else:
                print(f"Context:          {context['context_hash']}")
                print(
                    f"As of:            {context['as_of'].get('commit') or 'worktree'}"
                )
                print(f"Sources:          {len(context['source_object_ids'])}")
                print(f"Memory objects:   {len(context['memory_object_ids'])}")
                print(f"Negative memory:  {len(context['negative_knowledge_ids'])}")
                print(
                    "Belief conflicts: "
                    f"{len(context['belief_context']['conflict_sets'])}"
                )
                print(f"Complete:         {context['complete']}")
                for blocker in context["blockers"]:
                    print(f"  blocker: {_display_text(blocker)}")
            return 0 if context["complete"] else 1

        if args.command == "decide":
            from .research_policy import (
                decide_research_transition,
                record_research_transition_decision,
            )

            decision_args = {
                "event": args.event,
                "name": args.name,
                "state": args.state,
                "source_branch": args.source_branch,
                "competing_hypothesis": args.competing_hypothesis,
                "contradictory_evidence": args.contradictory_evidence,
                "protocol_change": args.protocol_change,
                "independent_replication": args.independent_replication,
            }
            checkpoint = None
            if args.record:
                recorded = record_research_transition_decision(
                    args.repo,
                    **decision_args,
                    actor_id=args.actor,
                )
                decision_payload = recorded["decision"]
                decision_object = recorded["decision_object"]
                context_object = recorded["context_object"]
                if not args.no_commit:
                    repository_root = Path(
                        repository_status(args.repo)["repository"]
                    ).resolve()
                    selected_paths = [
                        decision_object.path.relative_to(repository_root).as_posix()
                    ]
                    if context_object is not None:
                        selected_paths.append(
                            context_object.path.relative_to(repository_root).as_posix()
                        )
                        context_payload = load_research_object(
                            args.repo, context_object.object_id
                        )["payload"]
                        selected_paths.extend(
                            f".xscientist/objects/{item['kind']}/"
                            f"{item['object_id']}.json"
                            for item in context_payload.get("source_objects") or []
                        )
                    checkpoint = create_checkpoint(
                        args.repo,
                        stage="decision",
                        subject=f"adopt {args.event} transition decision",
                        only_paths=sorted(set(selected_paths)),
                    )
                payload = {
                    "decision": decision_payload,
                    "object": decision_object.to_dict(),
                    "context_object": (
                        context_object.to_dict() if context_object is not None else None
                    ),
                    "checkpoint": checkpoint.to_dict() if checkpoint else None,
                }
            else:
                if args.no_commit:
                    raise ResearchGitError("--no-commit requires --record")
                payload = decide_research_transition(args.repo, **decision_args)
            if args.as_json:
                _print_json(payload)
            else:
                decision = payload.get("decision") or payload
                print(f"Decision: {decision['decision_id']}")
                print(f"Event:    {decision['event']}")
                print(f"Branch:   {decision['branch']}")
                if payload.get("object"):
                    print(f"Recorded: {payload['object']['object_id']}")
                if payload.get("checkpoint"):
                    print(f"Commit:   {payload['checkpoint']['commit']}")
                for action in decision["actions"]:
                    print(f"{action['action']}: {_display_text(action['reason'])}")
                    for command in action["commands"]:
                        print(f"  {command}")
            return 0

        if args.command == "tree":
            from .research_policy import build_research_technology_tree

            payload = build_research_technology_tree(args.repo)
            if args.as_json:
                _print_json(payload)
            else:
                counts = payload["counts"]
                print(
                    "Technology tree: "
                    f"{counts['nodes']} objects, {counts['edges']} relations, "
                    f"{counts['branches']} research lines"
                )
                print(f"Integrity:       {payload['integrity']['ok']}")
                print(f"Open frontier:   {len(payload['frontier'])}")
                for item in payload["frontier"]:
                    print(
                        f"  {item['object_id']} {item['kind']} "
                        f"[{item['classification']}]"
                    )
            return 0 if payload["integrity"]["ok"] else 1

        if args.command == "dag":
            from .research_dag import build_research_dag, export_research_dag

            if args.output:
                exported = export_research_dag(
                    args.repo,
                    args.output,
                    ref=args.ref,
                    ara_roots=args.ara,
                    disclose_summaries=not args.metadata_only,
                )
                payload = exported["graph"]
            else:
                exported = None
                payload = build_research_dag(
                    args.repo,
                    ref=args.ref,
                    ara_roots=args.ara,
                    disclose_summaries=not args.metadata_only,
                )
            if args.as_json:
                _print_json(payload)
            else:
                print(
                    f"Scientific DAG: {len(payload['nodes'])} nodes, "
                    f"{len(payload['edges'])} relations"
                )
                print(f"DAG integrity:  {payload['integrity']['is_dag']}")
                print(f"Closure:        {payload['scientific_closure']['status']}")
                print(f"Verification:   {payload['proof_summary']}")
                ara_sources = [
                    source
                    for source in payload["sources"]
                    if source.get("name") != "research_vcs"
                ]
                print(f"ARA snapshots:  {len(ara_sources)}")
                for source in ara_sources:
                    manifest = source.get("manifest_integrity") or {}
                    graph_binding = source.get("graph_binding") or {}
                    location = source.get("repository_path") or "external"
                    print(
                        f"  {source['name']} [{manifest.get('state') or 'unknown'}] "
                        f"graph={graph_binding.get('state') or 'unknown'} "
                        f"bindings={source.get('binding_count', 0)} "
                        f"source={_display_text(location)}"
                    )
                for issue in payload["integrity"].get("issues") or []:
                    print(
                        f"  {issue.get('severity', 'warning')}: "
                        f"{issue.get('code', 'integrity_issue')} — "
                        f"{_display_text(issue.get('message') or '')}"
                    )
                if exported:
                    print(f"JSON:           {_display_path(exported['json'])}")
                    print(f"Browser:        {_display_path(exported['html'])}")
                else:
                    print(
                        "Write browser:  xscientist research dag --output research-dag"
                    )
            return 0 if payload["integrity"]["is_dag"] else 1

        if args.command == "adapter":
            from .research_adapters import (
                available_research_adapters,
                doctor_research_adapter,
                sync_research_repository,
            )

            if args.adapter_command == "list":
                payload = {"adapters": available_research_adapters()}
                if args.as_json:
                    _print_json(payload)
                else:
                    for item in payload["adapters"]:
                        print(
                            f"{item['name']:<20} {item['source']:<24} "
                            f"{_display_text(item['description'])}"
                        )
                return 0
            if args.adapter_command == "doctor":
                payload = doctor_research_adapter(args.name)
                if args.as_json:
                    _print_json(payload)
                else:
                    print(f"Adapter: {payload['adapter']['name']}")
                    print(f"Ready:   {payload['ok']}")
                    for error in payload["errors"]:
                        print(f"error: {_display_text(error)}", file=sys.stderr)
                return 0 if payload["ok"] else 1
            payload = sync_research_repository(
                args.repo,
                adapter_name=args.name,
                destination=args.dest,
                ref=args.ref,
                formats=args.formats
                or ["ro-crate", "prov-json", "cwl", "dvc", "mlflow"],
                include_payloads=args.include_payloads,
                options=_parse_assignments(args.option, label="adapter option"),
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Adapter: {_display_text(payload['adapter']['name'])}")
                print(f"Status:  {_display_text(payload['result'].get('status'))}")
                print(f"Receipt: {payload['receipt_hash']}")
            return 0

        if args.command == "fsck":
            payload = verify_research_repository(
                args.repo,
                commit=args.commit,
                verify_objects=not args.no_objects,
            )
            if args.as_json:
                _print_json(payload)
            else:
                verdict = "ok" if payload["ok"] else "failed"
                print(f"Research repository: {verdict}")
                print(f"Commit:              {payload['commit']}")
                checked = payload["checked"]
                print(
                    "Checked:             "
                    f"{checked['checkpoints']} checkpoints, "
                    f"{checked['pointers']} pointers, "
                    f"{checked['objects']} objects, "
                    f"{checked['ara_manifests']} ARA manifests"
                )
                for warning in payload["warnings"]:
                    print(f"warning: {_display_text(warning)}", file=sys.stderr)
                for error in payload["errors"]:
                    print(f"error: {_display_text(error)}", file=sys.stderr)
            return 0 if payload["ok"] else 1

        if args.command == "checkpoint":
            operation = commit_research_stage if args.staged else create_checkpoint
            checkpoint_kwargs = dict(
                repo=args.repo,
                stage=args.stage,
                subject=args.subject,
                summary=args.summary,
                status=args.status,
                actor=args.actor,
                nodes=args.node,
                claims=args.claim,
                ara_paths=args.ara,
                object_refs=args.object_ref,
                reproduce_command=args.reproduce,
            )
            if not args.staged:
                checkpoint_kwargs.update(
                    include=args.include,
                    commit=not args.no_commit,
                    allow_checkpoint_only=args.allow_checkpoint_only,
                )
            elif args.no_commit or args.allow_checkpoint_only or args.include:
                raise ResearchGitError(
                    "--staged cannot be combined with --no-commit, "
                    "--allow-checkpoint-only, or --include"
                )
            result = operation(**checkpoint_kwargs)
            payload = result.to_dict()
            if args.as_json:
                _print_json(payload)
            elif result.committed:
                print(
                    f"Checkpoint committed: {result.commit[:12]} {result.checkpoint_id}"
                )
                print(f"Stage: {args.stage}; files: {len(result.staged_paths)}")
            elif result.created:
                print(f"Checkpoint written: {_display_path(result.checkpoint_path)}")
            else:
                print(f"Checkpoint skipped: {result.reason}")
            if result.excluded_paths and not args.as_json:
                print("Excluded by safety policy:")
                for item in result.excluded_paths:
                    print(f"  - {item}")
            return 0

        if args.command == "record":
            if args.state in {"verified", "promoted"}:
                raise ResearchGitError(
                    "raw record cannot create verified or promoted objects; "
                    "use the evidence/review/claim/reproduce lifecycle commands"
                )
            semantic_profile = None
            if args.profile_file:
                profile_path = Path(args.profile_file).expanduser()
                if not profile_path.is_file():
                    raise ResearchGitError("semantic profile file was not found")
                try:
                    semantic_profile = json.loads(
                        profile_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError) as exc:
                    raise ResearchGitError(
                        "semantic profile file is invalid JSON"
                    ) from exc
                if not isinstance(semantic_profile, dict):
                    raise ResearchGitError("semantic profile must be a JSON object")
            result = record_research_object(
                args.repo,
                kind=args.kind,
                state=args.state,
                payload=_read_object_payload(args),
                relations=_parse_relations(args.relation),
                semantic_profile=semantic_profile,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                action = "Recorded" if result.created else "Already recorded"
                print(f"{action}: {result.object_id}")
                print(f"Kind/state: {result.kind} / {result.state}")
                print(f"Path:       {_display_path(result.path)}")
            return 0

        if args.command == "objects":
            if args.object_id:
                payload = load_research_object(args.repo, args.object_id)
            else:
                payload = list_research_objects(
                    args.repo,
                    kind=args.kind,
                    state=args.state,
                )
            if args.as_json or args.object_id:
                _print_json(payload)
            else:
                for item in payload:
                    print(
                        f"{item['object_id']} {item['kind']} "
                        f"[{item['state']}] {_display_text(_object_summary(item))}"
                    )
            return 0

        if args.command in {"stage", "add"}:
            result = research_stage(
                args.repo,
                args.paths,
                all_changes=args.all_changes,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Research stage: {len(result.paths)} path(s)")
                for path in result.paths:
                    print(f"  {path}")
                for item in result.excluded:
                    print(f"excluded: {_display_text(item)}", file=sys.stderr)
            return 0

        if args.command == "commit":
            if args.all_changes:
                research_stage(args.repo, all_changes=True)
            result = commit_research_stage(
                args.repo,
                stage=args.stage,
                subject=args.message,
                summary=args.summary,
                status=args.status,
                actor=args.actor,
            )
            if args.as_json:
                _print_json(result.to_dict())
            elif result.committed:
                print(
                    f"Research checkpoint: {result.checkpoint_id} "
                    f"files={len(result.staged_paths)}"
                )
            else:
                print(f"Research checkpoint skipped: {result.reason}")
            return 0

        if args.command == "unstage":
            result = research_unstage(
                args.repo,
                args.paths,
                all_paths=args.all_paths,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Research stage: {len(result.paths)} path(s)")
            return 0

        if args.command == "branch":
            if (args.delete or args.force_delete or args.move) and not args.name:
                raise ResearchGitError("branch maintenance requires a branch name")
            if args.delete or args.force_delete:
                payload = delete_research_branch(
                    args.repo,
                    args.name,
                    force=args.force_delete,
                )
            elif args.move:
                payload = rename_research_branch(args.repo, args.name, args.move)
            elif args.name:
                payload = create_research_branch(
                    args.repo,
                    args.name,
                    from_ref=args.from_ref,
                    switch=args.switch,
                )
            else:
                payload = list_research_branches(args.repo)
            if args.as_json:
                _print_json(payload)
            elif args.delete or args.force_delete:
                print(f"Research branch deleted: {payload['name']}")
            elif args.move:
                print(
                    f"Research branch renamed: {payload['old_name']} -> {payload['name']}"
                )
            elif args.name:
                marker = " and switched" if args.switch else ""
                print(f"Research branch created{marker}: {payload['name']}")
            else:
                for item in payload:
                    marker = "*" if item["current"] else " "
                    print(
                        f"{marker} {item['name']} "
                        f"{(item['checkpoint_id'] or '-')}: {item['subject']}"
                    )
            return 0

        if args.command == "switch":
            payload = switch_research_branch(args.repo, args.name)
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Research branch: {payload['name']}")
                print(f"Commit:          {payload['commit']}")
            return 0

        if args.command == "restore":
            payload = restore_research_paths(args.repo, args.source, args.paths)
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Restored from: {payload['source']}")
                for path in payload["paths"]:
                    print(f"  {path}")
                print("Review the working changes, then checkpoint them explicitly.")
            return 0

        if args.command == "revert":
            payload = revert_research_checkpoint(
                args.repo,
                args.commit,
                subject=args.message,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Reverted commit:    {payload['reverted']}")
                print(f"Backend revert:     {payload['revert_commit']}")
                print(
                    "Research checkpoint: "
                    f"{payload['checkpoint'].get('checkpoint_id') or '(not committed)'}"
                )
            return 0

        if args.command == "tag":
            payload = (
                create_research_tag(
                    args.repo,
                    args.name,
                    commit=args.commit,
                    annotation=args.annotation,
                )
                if args.name
                else list_research_tags(args.repo)
            )
            if args.as_json:
                _print_json(payload)
            elif args.name:
                print(
                    f"Research tag: {payload['name']} -> " f"{payload['checkpoint_id']}"
                )
            else:
                for item in payload:
                    print(f"{item['name']} {item['checkpoint_id'] or '-'}")
            return 0

        if args.command == "blame":
            payload = research_blame(
                args.repo,
                args.object_id,
                commit=args.commit,
            )
            if args.as_json:
                _print_json(payload)
            else:
                origin = payload["origin"]
                research_object = payload["object"]
                print(
                    f"{research_object['object_id']} "
                    f"{research_object['kind']} [{research_object['state']}]"
                )
                print(f"Origin checkpoint: {origin['checkpoint_id'] or '-'}")
                print(f"Origin commit:     {origin['commit']}")
                print(f"Subject:           {origin['subject']}")
                print(f"Outgoing links:    {len(payload['relations'])}")
                print(f"Incoming links:    {len(payload['related_by'])}")
            return 0

        if args.command == "merge":
            if args.preview:
                payload = preview_research_merge(args.repo, args.source)
            else:
                payload = merge_research_branch(
                    args.repo,
                    args.source,
                    subject=args.subject,
                    summary=args.summary,
                    actor=args.actor,
                    preserve_conflicts=args.preserve_conflicts,
                ).to_dict()
            if args.as_json:
                _print_json(payload)
            elif args.preview:
                verdict = "clean" if payload["clean"] else "blocked"
                print(f"Merge preflight: {verdict}")
                for conflict in payload["conflicts"]:
                    print(f"  {conflict['type']}: {_display_text(conflict['message'])}")
            else:
                print(
                    f"Merged research line: {payload['source']} -> {payload['target']}"
                )
                print(f"Checkpoint:           {payload['checkpoint_id']}")
                print(f"Commit:               {payload['commit']}")
                for object_id in payload.get("resolution_objects", []):
                    print(f"Contested hold gate:  {object_id}")
            return 0

        if args.command == "log":
            payload = research_log(args.repo, limit=args.limit)
            if args.as_json:
                _print_json(payload)
            else:
                for entry in payload:
                    stage = (entry["trailers"].get("Research-Stage") or ["-"])[0]
                    print(
                        f"{entry['short_commit']} {entry['authored_at']} "
                        f"[{stage}] {entry['subject']}"
                    )
            return 0

        if args.command == "trajectory":
            payload = research_trajectory(
                args.repo,
                limit=args.limit,
                ref=args.ref,
            )
            if args.as_json:
                _print_json(payload)
            else:
                scope = "complete" if payload["complete"] else "truncated"
                print(
                    f"Structured trajectory: {scope}; "
                    f"{payload['checkpoint_count']} checkpoints, "
                    f"{payload['object_count']} typed objects, "
                    f"{payload['object_transition_count']} object transitions"
                )
                print(f"Projection hash:      {payload['projection_hash']}")
                for entry in payload["entries"]:
                    print(
                        f"{entry['sequence']:04d} {entry['commit'][:12]} "
                        f"[{entry['stage']}/{entry['status']}] {entry['subject']}"
                    )
                    parent_commits = [
                        str(parent)[:12] for parent in entry["parent_commits"]
                    ]
                    print(
                        "     "
                        f"checkpoint={entry['checkpoint_hash']} "
                        f"parents={','.join(parent_commits) or '-'} "
                        f"actor={entry['actor'] or '-'}"
                    )
                    if entry.get("stage") == "revert":
                        print(
                            "     "
                            f"rollback={str(entry.get('reverts_commit') or '')[:12]} "
                            f"checkpoint={entry.get('reverts_checkpoint_hash')}"
                        )
                    for research_object in entry["objects"]:
                        print(
                            "     "
                            f"{research_object['object_id']} "
                            f"{research_object['kind']} "
                            f"[{research_object['state']}; "
                            f"{research_object['change']}]"
                        )
                if payload["boundary_parent_edges"]:
                    print("Boundary parent edges (projection is truncated):")
                    for edge in payload["boundary_parent_edges"]:
                        print(
                            "     "
                            f"{edge['parent_commit'][:12]} -> "
                            f"{edge['child_commit'][:12]} "
                            f"checkpoint={edge['parent_checkpoint_hash']}"
                        )
                if payload["boundary_rollback_edges"]:
                    print("Boundary rollback edges (projection is truncated):")
                    for edge in payload["boundary_rollback_edges"]:
                        print(
                            "     "
                            f"{edge['target_commit'][:12]} <- "
                            f"{edge['revert_commit'][:12]} "
                            f"checkpoint={edge['target_checkpoint_hash']}"
                        )
            return 0

        if args.command == "show":
            payload = show_checkpoint(args.repo, args.commit)
            if args.as_json:
                _print_json(payload)
            else:
                checkpoint = payload["checkpoint"]
                print(f"Commit:     {payload['commit']}")
                print(f"Checkpoint: {checkpoint.get('checkpoint_id')}")
                print(f"Stage:      {checkpoint.get('stage')}")
                print(f"State:      {checkpoint.get('status')}")
                print(f"Subject:    {checkpoint.get('subject')}")
                print(f"Summary:    {checkpoint.get('summary')}")
                for manifest in checkpoint.get("ara_manifests") or []:
                    print(f"ARA:        {manifest.get('path')}")
                    print(f"  manifest: {manifest.get('manifest_hash')}")
                    if manifest.get("exploration_graph_hash"):
                        print(f"  graph:    {manifest['exploration_graph_hash']}")
                if (checkpoint.get("reproduce") or {}).get("command"):
                    print(f"Reproduce:  {checkpoint['reproduce']['command']}")
            return 0

        if args.command == "diff":
            payload = research_diff(
                args.repo,
                args.before,
                args.after,
                deep=args.deep,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"{payload['before']}..{payload['after']}")
                for line in payload["changes"]:
                    print(line)
                if payload["stat"]:
                    print(payload["stat"])
                semantic = payload["semantic"]
                print(
                    "Scientific checkpoints: "
                    f"{semantic['before_checkpoint']['checkpoint_id']} -> "
                    f"{semantic['after_checkpoint']['checkpoint_id']}"
                )
                print(
                    "Claims: "
                    f"+{len(semantic['claims']['added'])} "
                    f"-{len(semantic['claims']['removed'])}; "
                    "nodes: "
                    f"+{len(semantic['nodes']['added'])} "
                    f"-{len(semantic['nodes']['removed'])}; "
                    "objects: "
                    f"+{len(semantic['objects']['added'])} "
                    f"-{len(semantic['objects']['removed'])}"
                )
                typed = semantic["research_objects"]
                print(
                    "Typed research objects: "
                    f"+{len(typed['added'])} "
                    f"-{len(typed['removed'])}; "
                    f"relations +{len(typed['relations']['added'])} "
                    f"-{len(typed['relations']['removed'])}"
                )
                manifests = semantic["ara_manifests"]
                print(
                    "ARA snapshots: "
                    f"+{len(manifests['added'])} "
                    f"-{len(manifests['removed'])} "
                    f"~{len(manifests['changed'])}"
                )
                for change in manifests["changed"]:
                    print(
                        f"  {change['path']}: " f"{', '.join(change['changed_fields'])}"
                    )
                if args.deep:
                    print(
                        "Structured field changes: "
                        f"{len(semantic['structured_changes'])}"
                    )
            return 0

        if args.command == "object" and args.object_command == "add":
            result = add_research_object(
                args.repo,
                args.source,
                logical_path=args.logical_path,
                media_type=args.media_type,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Object:  {result.object_hash}")
                print(f"Pointer: {_display_path(result.pointer_path)}")
                print(f"Store:   {_display_path(result.store_path)}")
                print("Run `xscientist research checkpoint` to commit the pointer.")
            return 0

        if args.command == "export":
            from .research_interop import INTEROP_FORMATS, export_research_interop

            payload = export_research_interop(
                args.repo,
                args.dest,
                ref=args.ref,
                formats=args.formats or INTEROP_FORMATS,
                include_payloads=args.include_payloads,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Research export: {_display_path(payload['destination'])}")
                print(f"Commit:          {payload['repository_commit']}")
                print(f"Checkpoint:      {payload['checkpoint_id']}")
                print(f"Objects:         {payload['object_count']}")
                print(f"Formats:         {', '.join(payload['formats'])}")
                print(f"Export hash:     {payload['export_hash']}")
            return 0

        if args.command == "bundle":
            if args.action == "create":
                if not args.dest:
                    raise ResearchGitError("bundle create requires --dest")
                payload = create_research_bundle(
                    args.repo,
                    args.dest,
                    profile=args.profile,
                    allow_incomplete=args.allow_incomplete,
                )
            elif args.action == "verify":
                if not args.bundle_path:
                    raise ResearchGitError("bundle verify requires a bundle path")
                payload = verify_research_bundle(args.bundle_path)
            else:
                if not args.bundle_path or not args.dest:
                    raise ResearchGitError(
                        "bundle restore requires a bundle path and --dest"
                    )
                payload = restore_research_bundle(args.bundle_path, args.dest)
            if args.as_json:
                _print_json(payload)
            elif args.action == "verify":
                print(f"Bundle:   {_display_path(payload['bundle'])}")
                print(f"Valid:    {payload['ok']}")
                for warning in payload["warnings"]:
                    print(f"warning: {_display_text(warning)}", file=sys.stderr)
                for error in payload["errors"]:
                    print(f"error: {_display_text(error)}", file=sys.stderr)
            elif args.action == "restore":
                print(f"Bundle:     {_display_path(payload['bundle'])}")
                print(f"Repository: {_display_path(payload['repository'])}")
                print(f"HEAD:       {payload['commit']}")
                print(f"Objects:    {payload['objects_restored']}")
            else:
                print(f"Bundle:   {_display_path(payload['destination'])}")
                print(f"Profile:  {payload['profile']}")
                print(f"Complete: {payload['complete']}")
                print(f"HEAD:     {payload['repository_head']}")
            if args.action == "verify":
                return 0 if payload["ok"] else 1
            return 0

        if args.command == "reproduce":
            if not args.record and (
                args.verified or args.reproduces or args.verifier or args.no_commit
            ):
                raise ResearchGitError(
                    "--verified, --reproduces, --verifier, and --no-commit require --record"
                )
            if args.record and not args.reproduces:
                raise ResearchGitError(
                    "--record requires at least one --reproduces object; for the "
                    "current claim use `--reproduces @latest:claim`"
                )
            if args.verified and not str(args.verifier or "").strip():
                raise ResearchGitError("--verified requires --verifier")
            payload = reproduce_checkpoint(
                args.repo,
                commit=args.commit,
                destination=args.dest,
                execute=args.execute,
                timeout_seconds=args.timeout,
                environment_policy=args.environment_policy,
            )
            if args.record:
                from .research_lifecycle import ResearchLifecycle

                recorded = ResearchLifecycle(args.repo).reproduction(
                    payload["receipt"],
                    reproduces=args.reproduces,
                    verifier_id=args.verifier,
                    verified=args.verified,
                    commit=not args.no_commit,
                )
                payload["recorded_reproduction"] = {
                    "object": recorded["reproduction"].to_dict(),
                    "checkpoint": (
                        recorded["checkpoint"].to_dict()
                        if recorded["checkpoint"] is not None
                        else None
                    ),
                }
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Commit:           {payload['commit']}")
                print(f"Checkpoint:       {payload['checkpoint']['checkpoint_id']}")
                print(f"Objects complete: {payload['objects_complete']}")
                print(f"Environment:      {payload['environment']['matches']}")
                print(
                    "Command:          "
                    f"{_display_text(payload['command'] or '(not declared)')}"
                )
                if payload.get("limitation"):
                    print(f"Limitation:       {_display_text(payload['limitation'])}")
                if payload.get("next_action"):
                    print(f"Next:             {_display_text(payload['next_action'])}")
                if (
                    args.environment_policy == "warn"
                    and payload["environment"]["mismatches"]
                ):
                    for mismatch in payload["environment"]["mismatches"]:
                        print(
                            f"warning: environment mismatch: {mismatch['field']}",
                            file=sys.stderr,
                        )
                if payload.get("worktree"):
                    print(f"Worktree:         {_display_path(payload['worktree'])}")
                if payload.get("executed"):
                    print(f"Return code:      {payload['returncode']}")
                print(f"Receipt:          {payload['receipt']['receipt_id']}")
                if payload.get("receipt_path"):
                    print(f"Receipt path:     {payload['receipt_path']}")
                if payload.get("recorded_reproduction"):
                    print(
                        "Recorded object:   "
                        f"{payload['recorded_reproduction']['object']['object_id']}"
                    )
            return int(payload.get("returncode") or 0) if args.execute else 0
    except ResearchGitError as exc:
        message = redact_sensitive_text(str(exc))
        if getattr(args, "as_json", False):
            _print_json(
                {
                    "schema_version": "xscientist.error.v1",
                    "ok": False,
                    "error": {
                        "category": "research_vcs_error",
                        "command": args.command,
                        "message": message,
                    },
                }
            )
        else:
            print(f"research vcs error: {message}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
