"""Prepare and execute a deterministic refinement smoke example without a model."""

from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path


def _evaluate() -> None:
    root = Path(os.environ["XSCIENTIST_ARTIFACT_ROOT"])
    holdout = os.environ["XSCIENTIST_TASK_ID"] == "holdout"
    parser = runpy.run_path(str(root / "parse.py"))
    estimator = runpy.run_path(str(root / "estimate.py"))
    rows = ["1200", "2400"] if holdout else ["1,000", "2,000"]
    parsed = []
    for value in rows:
        try:
            parsed.append(parser["parse"](value))
        except ValueError:
            pass
    estimate = estimator["estimate"]([2, 4, 6] if holdout else [1, 3, 5])
    score = (len(parsed) / len(rows) + float(estimate == (4 if holdout else 3))) / 2
    events = [
        {
            "event_id": "call-" + name,
            "change_id": module["CHANGE_ID"],
            "kind": "activated",
            "detail": "Evaluator invoked " + name,
        }
        for name, module in (("parse", parser), ("estimate", estimator))
        if module.get("CHANGE_ID")
    ]
    print(
        json.dumps(
            {
                "metrics": {
                    "objective_score": score,
                    "reproducibility_rate": 1.0,
                    "false_discovery_rate": 0.0,
                    "cost_per_task": 0.0,
                    "latency_seconds": 0.0,
                },
                "safety_pass": True,
                "integrity_pass": True,
                "reproducibility_pass": True,
                "behavior_evidence": events,
            }
        )
    )


def prepare(root: Path) -> tuple[dict, dict]:
    """Write reproducible proposal inputs into a new output directory."""
    from ai_scientist.protocol.canonical_json import canonical_content_hash
    from ai_scientist.utils.evolution_artifacts import (
        build_evolution_candidate_from_sources,
    )
    from ai_scientist.utils.evolution_runtime import BENCHMARK_SUITE_SCHEMA
    from ai_scientist.utils.harness_refinement import SPEC_SCHEMA
    from ai_scientist.utils.science_constitution import build_science_constitution

    root.mkdir(parents=True, exist_ok=False)
    constitution = build_science_constitution(project_name="harness-refinement-demo")
    files = {
        "parse.py": "CHANGE_ID = None\ndef parse(value):\n    return int(value)\n",
        "estimate.py": "CHANGE_ID = None\ndef estimate(values):\n    return sum(values)\n",
    }
    edits = {
        "inv": {
            "parse.py": "CHANGE_ID = 'inv'\ndef parse(value):\n    return int(value.replace(',', ''))\n"
        },
        "dom": {
            "estimate.py": "CHANGE_ID = 'dom'\ndef estimate(values):\n    return sum(values) / len(values)\n"
        },
    }
    for name, content in [
        ("base", files),
        *[(key, {**files, **value}) for key, value in edits.items()],
    ]:
        for filename, text in content.items():
            target = root / name / "failure_recovery" / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
    proposals = []
    for pid, dimension in (("inv", "invariant"), ("dom", "domain")):
        build = build_evolution_candidate_from_sources(
            constitution=constitution,
            base_root=root / "base",
            candidate_root=root / pid,
            store_root=root / "cas",
            candidate_id=pid,
            component_type="failure_recovery",
            base_version="1.0",
            candidate_version="1.1-" + pid,
            proposed_by="service:demo",
            change_summary="Demonstrate one observable repair.",
            change_scope=["failure_recovery/" + name for name in edits[pid]],
            applicability_domains=["numeric-input-smoke"],
            failure_taxonomy_refs=["failure:" + pid],
            ablation_dimensions=[pid],
        )
        proposals.append(
            {
                "proposal_id": pid,
                "dimension": dimension,
                "candidate": build["candidate"],
                "cue": "Numeric input parsing and an arithmetic mean calculation",
                "hypothesis": (
                    "Comma-aware parsing recovers valid rows."
                    if pid == "inv"
                    else "Divide the sum by the count to compute the mean."
                ),
                "falsifier": "The procedure is unused, regresses, or fails the held-out inputs.",
            }
        )
    script = Path(__file__).resolve()
    script_hash = canonical_content_hash(script.read_text(encoding="utf-8"))

    def suite(task: str, split: str) -> dict:
        return {
            "schema_version": BENCHMARK_SUITE_SCHEMA,
            "benchmark_id": task,
            "producer_stack_id": "service:demo-proposals",
            "producer_stack_hash": canonical_content_hash(proposals),
            "evaluator_stack_id": "service:demo-evaluator",
            "evaluator_stack_hash": script_hash,
            "evaluator_id": "service:demo-evaluator",
            "tasks": [
                {
                    "task_id": task,
                    "task_hash": canonical_content_hash(
                        {"task": task, "script_hash": script_hash}
                    ),
                    "split": split,
                    "evaluation_layer": "public" if split == "public" else "sealed",
                    "frozen_before_candidate": True,
                    "command": [sys.executable, str(script), "--evaluate"],
                }
            ],
        }

    spec = {
        "schema_version": SPEC_SCHEMA,
        "proposals": proposals,
        "development_suite": suite("development", "public"),
        "holdout_suite": suite("holdout", "hidden"),
    }
    for name, payload in (("spec.json", spec), ("constitution.json", constitution)):
        (root / name).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return spec, constitution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluate", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.evaluate:
        _evaluate()
        return 0
    if args.output is None:
        parser.error("--output is required")
    from ai_scientist.utils.harness_refinement import run_harness_refinement

    root = args.output.expanduser().resolve()
    spec, constitution = prepare(root)
    report = run_harness_refinement(
        spec,
        constitution=constitution,
        artifact_store=root / "cas",
        allow_execution=True,
    )
    output = root / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "executions": report["execution_count"],
                "report": str(output),
            }
        )
    )
    return 0 if report["decision"] == "shadow_candidate_selected" else 3


if __name__ == "__main__":
    raise SystemExit(main())
