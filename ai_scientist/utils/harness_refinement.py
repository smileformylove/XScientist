"""Opt-in, evidence-guided composition of immutable shadow candidates.

Proposal authors supply atomic INV/DOM candidates. Only public development
probes guide composition; a frozen winner gets one held-out evaluation. These
local experiments never grant production or persistent-learning authority.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Mapping

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_artifacts import (
    build_evolution_candidate_from_sources,
    materialize_evolution_artifact,
    verify_evolution_artifact,
)
from ai_scientist.utils.evolution_gate import (
    DEFAULT_METRIC_SPECS,
    MUTABLE_COMPONENT_SCOPES,
    validate_evolution_candidate,
)
from ai_scientist.utils.evolution_runtime import (
    EvolutionRuntimeError,
    run_shadow_benchmark,
    validate_benchmark_suite,
)

SPEC_SCHEMA = "xscientist.harness-refinement-spec.v1"
REPORT_SCHEMA = "xscientist.harness-refinement.v1"
_MAX_METRIC = 1.0e12
_DEFAULT_POLICY = {
    "repeats": 2,
    "minimum_improvement": 0.01,
    "max_executions": 128,
    "max_cost": 100.0,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvolutionRuntimeError(message)


def _policy(raw: Any) -> dict[str, Any]:
    _require(isinstance(raw, dict), "refinement policy must be an object")
    _require(not set(raw) - set(_DEFAULT_POLICY), "unsupported refinement policy field")
    policy = {**_DEFAULT_POLICY, **raw}
    for name, lower, upper in (("repeats", 2, 8), ("max_executions", 2, 1024)):
        value = policy[name]
        _require(type(value) is int and lower <= value <= upper, f"invalid {name}")
    for name in ("minimum_improvement", "max_cost"):
        value = policy[name]
        _require(
            type(value) in (int, float) and 0 < value <= _MAX_METRIC,
            f"{name} must be positive and at most {_MAX_METRIC:g}",
        )
    return policy


def _manifest(store: str | Path, digest: str, component: str) -> dict[str, Any]:
    result = verify_evolution_artifact(store, digest)
    _require(result["ok"], "refinement artifact failed verification")
    manifest = result["manifest"]
    prefix = MUTABLE_COMPONENT_SCOPES[component].rstrip("/")
    _require(
        manifest.get("component_type") == component
        and manifest.get("logical_root") == prefix,
        "artifact component binding mismatch",
    )
    for item in manifest["entries"]:
        path = PurePosixPath(item["path"])
        _require(
            not path.is_absolute()
            and ".." not in path.parts
            and item["path"].startswith(prefix + "/"),
            "artifact path escapes the mutable component",
        )
    return manifest


def _delta(base: Mapping[str, Any], changed: Mapping[str, Any]) -> list[str]:
    left = {row["path"]: row for row in base["entries"]}
    right = {row["path"]: row for row in changed["entries"]}
    return sorted(
        path for path in left.keys() | right.keys() if left.get(path) != right.get(path)
    )


def _validate(
    spec: Mapping[str, Any], constitution: dict[str, Any], store: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _require(isinstance(spec, dict), "refinement spec must be an object")
    _require(
        not set(spec)
        - {
            "schema_version",
            "proposals",
            "development_suite",
            "holdout_suite",
            "policy",
        },
        "unsupported refinement spec field",
    )
    _require(
        spec.get("schema_version") == SPEC_SCHEMA, "invalid refinement spec schema"
    )
    policy = _policy(spec.get("policy", {}))
    proposals = deepcopy(spec.get("proposals"))
    _require(
        isinstance(proposals, list) and 1 <= len(proposals) <= 8,
        "provide 1-8 proposals",
    )
    ids: set[str] = set()
    bindings: set[tuple[str, str, str]] = set()
    artifacts: set[str] = set()
    for proposal in proposals:
        _require(
            isinstance(proposal, dict)
            and set(proposal)
            == {
                "proposal_id",
                "dimension",
                "candidate",
                "hypothesis",
                "falsifier",
                "cue",
            },
            "invalid proposal fields",
        )
        pid = proposal["proposal_id"]
        _require(
            isinstance(pid, str)
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", pid) is not None
            and pid not in ids
            and pid != "composition",
            "proposal_id must be unique and bounded; composition is reserved",
        )
        ids.add(pid)
        _require(
            proposal["dimension"] in {"invariant", "domain"},
            "invalid proposal dimension",
        )
        for key in ("hypothesis", "falsifier", "cue"):
            value = proposal[key]
            _require(
                isinstance(value, str) and 0 < len(value.strip()) <= 2048,
                f"invalid {key}",
            )
        candidate = proposal["candidate"]
        _require(isinstance(candidate, dict), "proposal candidate must be an object")
        check = validate_evolution_candidate(candidate, constitution=constitution)
        _require(
            check["ok"], "invalid proposal candidate: " + ", ".join(check["errors"])
        )
        component = candidate["component_type"]
        base = _manifest(store, candidate["base_artifact_hash"], component)
        changed = _manifest(store, candidate["candidate_artifact_hash"], component)
        paths = _delta(base, changed)
        scopes = candidate["change_scope"]
        _require(
            isinstance(scopes, list)
            and all(isinstance(scope, str) for scope in scopes),
            "change_scope must be an array of paths",
        )
        _require(
            paths
            and all(
                any(
                    path == scope or path.startswith(scope.rstrip("/") + "/")
                    for scope in scopes
                )
                for path in paths
            ),
            "proposal contains undeclared changes",
        )
        _require(
            all(
                any(
                    path == scope or path.startswith(scope.rstrip("/") + "/")
                    for path in paths
                )
                for scope in scopes
            ),
            "proposal contains unused change scopes",
        )
        proposal["changed_paths"] = paths
        bindings.add(
            (component, candidate["base_artifact_hash"], candidate["base_version"])
        )
        artifacts.add(candidate["candidate_artifact_hash"])
    _require(
        len(bindings) == 1, "proposals must share one component, baseline and version"
    )
    _require(
        len(artifacts) == len(proposals),
        "duplicate candidate artifacts are not independent proposals",
    )
    suites: list[dict[str, Any]] = []
    for name in ("development_suite", "holdout_suite"):
        suite = spec.get(name)
        if name == "holdout_suite" and suite is None:
            continue
        _require(isinstance(suite, dict), f"{name} must be a benchmark suite")
        validation = validate_benchmark_suite(suite)
        _require(
            validation["ok"], f"invalid {name}: " + ", ".join(validation["errors"])
        )
        _require(len(suite["tasks"]) <= 16, "at most 16 tasks per suite")
        for task in suite["tasks"]:
            if name == "development_suite":
                _require(
                    task["split"] == "public" and task["evaluation_layer"] == "public",
                    "development probes must be public; hidden feedback cannot guide composition",
                )
            else:
                _require(
                    task["split"] == "hidden"
                    and task["evaluation_layer"] in {"sealed", "prospective"}
                    and task.get("frozen_before_candidate") is True,
                    "holdout tasks must be hidden and frozen before candidates",
                )
            _require(
                not any(
                    str(key).startswith("XSCIENTIST_")
                    for key in task.get("environment", {})
                ),
                "task environment cannot override runtime-owned XSCIENTIST variables",
            )
        suites.append(suite)
    if len(suites) == 2:
        dev, holdout = suites
        for field in (
            "producer_stack_id",
            "producer_stack_hash",
            "evaluator_stack_id",
            "evaluator_stack_hash",
        ):
            _require(
                dev[field] == holdout[field],
                "development and holdout stacks must match",
            )
        for field in ("task_id", "task_hash"):
            _require(
                not {t[field] for t in dev["tasks"]}
                & {t[field] for t in holdout["tasks"]},
                "development and holdout tasks must be disjoint",
            )
    return proposals, policy


def _assessment(
    reports: list[dict[str, Any]], change_ids: list[str], policy: dict[str, Any]
) -> dict[str, Any]:
    reasons: set[str] = set()
    evidence: list[dict[str, str]] = []
    samples = [sample for report in reports for sample in report["samples"]]
    if any(report["status"] != "completed" for report in reports) or not samples:
        reasons.add("execution_failed")
    for report in reports:
        for receipt in report["run_receipts"]:
            if receipt["variant"] != "candidate":
                continue
            activated = set()
            for event in receipt.get("behavior_evidence", []):
                if event["change_id"] in change_ids:
                    evidence.append({"run_hash": receipt["run_hash"], **event})
                    if event["kind"] == "activated":
                        activated.add(event["change_id"])
                    elif event["kind"] == "counterexample":
                        reasons.add("mechanism_counterexample")
            if activated != set(change_ids):
                reasons.add("activation_not_observed")
    deltas: list[float] = []
    for sample in samples:
        if not all(
            sample.get(key) is True
            for key in ("safety_pass", "integrity_pass", "reproducibility_pass")
        ):
            reasons.add("scientific_gate_failed")
        for name, metric in DEFAULT_METRIC_SPECS.items():
            base = sample["baseline"].get(name)
            changed = sample["candidate"].get(name)
            if (
                base is None
                or changed is None
                or max(abs(base), abs(changed)) > _MAX_METRIC
                or (
                    name in {"cost_per_task", "latency_seconds"}
                    and min(base, changed) < 0
                )
                or (
                    name in {"false_discovery_rate", "reproducibility_rate"}
                    and not (0 <= base <= 1 and 0 <= changed <= 1)
                )
            ):
                reasons.add("required_metric_missing_or_invalid")
                continue
            delta = (changed - base) * (1 if metric["direction"] == "higher" else -1)
            if delta < -metric["max_mean_regression"]:
                reasons.add("regression:" + name)
            if name == "objective_score":
                deltas.append(delta)
    # Repeated baseline draws are an explicit control for lucky candidate runs.
    for task_id in {sample["task_id"] for sample in samples}:
        rows = [row for row in samples if row["task_id"] == task_id]
        if len(rows) != policy["repeats"]:
            reasons.add("incomplete_repeats")
        if all(
            "objective_score" in row[variant]
            and abs(row[variant]["objective_score"]) <= _MAX_METRIC
            for row in rows
            for variant in ("baseline", "candidate")
        ):
            if (
                mean(row["candidate"]["objective_score"] for row in rows)
                - max(row["baseline"]["objective_score"] for row in rows)
                < policy["minimum_improvement"]
            ):
                reasons.add("does_not_beat_repeated_baseline")
    return {
        "eligible": not reasons,
        "reasons": sorted(reasons),
        "mean_improvement": mean(deltas) if deltas else None,
        "evidence": evidence,
    }


def _compose(
    selected: list[dict[str, Any]],
    constitution: dict[str, Any],
    store: str | Path,
    evidence_hashes: list[str],
) -> dict[str, Any]:
    first = selected[0]["candidate"]
    with tempfile.TemporaryDirectory(prefix="xscientist-compose-") as raw:
        root = Path(raw)
        base, combined = root / "base", root / "combined"
        for target in (base, combined):
            materialize_evolution_artifact(store, first["base_artifact_hash"], target)
        for index, proposal in enumerate(selected):
            source = root / f"proposal-{index}"
            materialize_evolution_artifact(
                store, proposal["candidate"]["candidate_artifact_hash"], source
            )
            for path in proposal["changed_paths"]:
                if not (source / path).is_file():
                    (combined / path).unlink()
            for path in proposal["changed_paths"]:
                target, changed = combined / path, source / path
                if changed.is_file():
                    if target.is_dir():
                        shutil.rmtree(target)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(changed, target)
        return build_evolution_candidate_from_sources(
            constitution=constitution,
            base_root=base,
            candidate_root=combined,
            store_root=store,
            candidate_id="harness-composition-"
            + canonical_content_hash(evidence_hashes)[7:23],
            component_type=first["component_type"],
            base_version=first["base_version"],
            candidate_version=first["base_version"] + ".harness-composed",
            proposed_by="service:harness-refinement",
            change_summary="Compose disjoint, observed shadow mechanisms.",
            change_scope=[
                path for proposal in selected for path in proposal["changed_paths"]
            ],
            applicability_domains=sorted(
                set.intersection(
                    *[
                        set(proposal["candidate"]["applicability_domains"])
                        for proposal in selected
                    ]
                )
            ),
            failure_taxonomy_refs=sorted(
                {
                    ref
                    for p in selected
                    for ref in p["candidate"]["failure_taxonomy_refs"]
                }
            ),
            ablation_dimensions=sorted(
                {ref for p in selected for ref in p["candidate"]["ablation_dimensions"]}
            ),
            provenance_hashes=[
                *evidence_hashes,
                *[p["candidate"]["candidate_hash"] for p in selected],
            ],
            risk_tier=max(
                (p["candidate"]["risk_tier"] for p in selected),
                key={"low": 0, "moderate": 1, "high": 2}.get,
            ),
        )["candidate"]


def run_harness_refinement(
    spec: dict[str, Any],
    *,
    constitution: dict[str, Any],
    artifact_store: str | Path,
    allow_execution: bool = False,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Probe supplied proposals, compose supported changes, then test a frozen winner."""
    spec = deepcopy(spec)
    proposals, policy = _validate(spec, constitution, artifact_store)
    _require(
        type(timeout_seconds) is int and timeout_seconds > 0, "timeout must be positive"
    )
    development, holdout = spec["development_suite"], spec.get("holdout_suite")
    composition_slots = int(
        {p["dimension"] for p in proposals} == {"invariant", "domain"}
    )
    planned = (
        2
        * policy["repeats"]
        * (
            (len(proposals) + composition_slots) * len(development["tasks"])
            + (len(holdout["tasks"]) if holdout else 0)
        )
    )
    _require(
        planned <= policy["max_executions"], "planned probes exceed max_executions"
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "spec_hash": canonical_content_hash(spec),
        "policy": policy,
        "spec": spec,
        "status": "planned",
        "decision": "keep_base",
        "production_promotion_allowed": False,
        "persistent_learning_allowed": False,
        "baseline_artifact_hash": proposals[0]["candidate"]["base_artifact_hash"],
        "planned_max_executions": planned,
        "execution_count": 0,
        "observed_cost": 0.0,
        "cost_unit": "evaluator_reported_cost_per_task",
        "accounting": {
            "baseline_executions": 0,
            "candidate_executions": 0,
            "baseline_reported_cost": 0.0,
            "candidate_reported_cost": 0.0,
            "process_seconds": 0.0,
            "proposal_generation_cost_observed": False,
        },
        "trials": [],
        "composition": {"status": "not_attempted"},
        "development_winner": None,
        "selected_candidate": None,
        "selection_commitment_hash": None,
        "holdout": None,
        "experience": [],
        "stop_reason": None,
    }
    if not allow_execution:
        report["report_hash"] = canonical_content_hash(report)
        return report

    def probe(
        candidate: dict[str, Any], suite: dict[str, Any], ids: list[str]
    ) -> dict[str, Any]:
        runs: list[dict[str, Any]] = []
        for task in suite["tasks"]:
            for repeat in range(policy["repeats"]):
                if report["stop_reason"]:
                    break
                if report["execution_count"] + 2 > policy["max_executions"]:
                    report["stop_reason"] = "execution_budget_exhausted"
                    break
                one = deepcopy(suite)
                one["tasks"] = [deepcopy(task)]
                one["tasks"][0].setdefault("environment", {})[
                    "XSCIENTIST_REFINEMENT_REPEAT"
                ] = str(repeat)
                result = run_shadow_benchmark(
                    one,
                    candidate,
                    artifact_store=artifact_store,
                    allow_execution=True,
                    timeout_seconds=timeout_seconds,
                )
                runs.append(result)
                report["execution_count"] += len(result["run_receipts"])
                for receipt in result["run_receipts"]:
                    variant = receipt["variant"]
                    report["accounting"][variant + "_executions"] += 1
                    report["accounting"]["process_seconds"] += receipt[
                        "duration_seconds"
                    ]
                    cost = receipt.get("metrics", {}).get("cost_per_task")
                    if cost is None or not 0 <= cost <= _MAX_METRIC:
                        report["stop_reason"] = "cost_unobserved_or_invalid"
                    else:
                        report["observed_cost"] += cost
                        report["accounting"][variant + "_reported_cost"] += cost
                    if receipt.get("status") == "completed" and not all(
                        receipt.get(key) is True
                        for key in (
                            "safety_pass",
                            "integrity_pass",
                            "reproducibility_pass",
                        )
                    ):
                        report["stop_reason"] = "scientific_gate_failed"
                if report["observed_cost"] >= policy["max_cost"]:
                    report["stop_reason"] = "cost_budget_exhausted"
        assessment = _assessment(runs, ids, policy)
        if (
            len(runs) != len(suite["tasks"]) * policy["repeats"]
            or report["stop_reason"]
        ):
            assessment["eligible"] = False
            assessment["reasons"] = sorted(
                set(assessment["reasons"])
                | {report["stop_reason"] or "incomplete_probes"}
            )
        return {"candidate": candidate, "change_ids": ids, "runs": runs, **assessment}

    eligible: list[dict[str, Any]] = []
    for proposal in proposals:
        if report["stop_reason"]:
            break
        trial = {
            "proposal_id": proposal["proposal_id"],
            "dimension": proposal["dimension"],
            **probe(proposal["candidate"], development, [proposal["proposal_id"]]),
        }
        report["trials"].append(trial)
        if trial["eligible"]:
            eligible.append(trial)
    selected: list[dict[str, Any]] = []
    for dimension in ("invariant", "domain"):
        options = [trial for trial in eligible if trial["dimension"] == dimension]
        if options:
            best = max(options, key=lambda t: (t["mean_improvement"], t["proposal_id"]))
            selected.append(
                next(p for p in proposals if p["proposal_id"] == best["proposal_id"])
            )
    if len(selected) == 2 and not report["stop_reason"]:
        paths = [p["changed_paths"] for p in selected]
        conflict = any(
            a == b or a.startswith(b + "/") or b.startswith(a + "/")
            for a in paths[0]
            for b in paths[1]
        )
        domains = set(selected[0]["candidate"]["applicability_domains"]) & set(
            selected[1]["candidate"]["applicability_domains"]
        )
        if conflict or not domains:
            report["composition"] = {
                "status": "skipped",
                "reason": "overlapping_changes" if conflict else "disjoint_domains",
            }
        else:
            try:
                candidate = _compose(
                    selected,
                    constitution,
                    artifact_store,
                    [run["report_hash"] for trial in eligible for run in trial["runs"]],
                )
            except (OSError, ValueError, EvolutionRuntimeError) as exc:
                report["composition"] = {
                    "status": "failed",
                    "error_category": type(exc).__name__,
                    "error_hash": canonical_content_hash(str(exc)),
                }
                report["stop_reason"] = "composition_failed"
            else:
                trial = {
                    "proposal_id": "composition",
                    "dimension": "composition",
                    **probe(
                        candidate, development, [p["proposal_id"] for p in selected]
                    ),
                }
                report["trials"].append(trial)
                report["composition"] = {
                    "status": "probed",
                    "candidate_hash": candidate["candidate_hash"],
                }
                if trial["eligible"]:
                    eligible.append(trial)
    if eligible and not report["stop_reason"]:
        winner = max(eligible, key=lambda t: (t["mean_improvement"], t["proposal_id"]))
        report["development_winner"] = winner["candidate"]["candidate_hash"]
        report["selection_commitment_hash"] = canonical_content_hash(
            {
                "candidate_hash": report["development_winner"],
                "policy": policy,
                "development_report_hashes": [
                    run["report_hash"] for t in report["trials"] for run in t["runs"]
                ],
            }
        )
        if holdout:
            report["holdout"] = probe(
                winner["candidate"], holdout, winner["change_ids"]
            )
            if report["holdout"]["eligible"] and not report["stop_reason"]:
                report["selected_candidate"] = winner["candidate"]
                report["decision"] = "shadow_candidate_selected"
    for proposal in proposals:
        trial = next(
            (
                t
                for t in report["trials"]
                if t["proposal_id"] == proposal["proposal_id"]
            ),
            None,
        )
        holdout_evidence = report["holdout"] or {}
        participated = proposal["proposal_id"] in holdout_evidence.get("change_ids", [])
        report["experience"].append(
            {
                "status": "quarantined",
                "cue": proposal["cue"],
                "lesson": proposal["hypothesis"],
                "falsifier": proposal["falsifier"],
                "proposal_id": proposal["proposal_id"],
                "applicability_domains": proposal["candidate"]["applicability_domains"],
                "artifact_hash": proposal["candidate"]["candidate_artifact_hash"],
                "evidence": trial["evidence"] if trial else [],
                "counterexamples": (trial["reasons"] if trial else ["not_executed"])
                + (
                    [
                        "holdout:" + reason
                        for reason in holdout_evidence.get("reasons", [])
                    ]
                    if participated
                    else []
                ),
                "holdout_report_hashes": (
                    [
                        r["report_hash"]
                        for r in (report["holdout"] or {}).get("runs", [])
                    ]
                    if participated
                    else []
                ),
            }
        )
    report["status"] = "halted" if report["stop_reason"] else "completed"
    report["report_hash"] = canonical_content_hash(report)
    return report
