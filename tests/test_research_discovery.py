from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from ai_scientist.protocol.research_vcs import research_payload_issues
from xscientist import ResearchRepository
from xscientist.research_cli import main as research_main
from xscientist.research_commands import save_claim
from xscientist.research_context import record_research_context_snapshot
from xscientist.research_discovery import (
    assess_generalization,
    build_discovery_contract,
    save_discovery_contract,
    save_generalization_assessment,
)


def _spec() -> dict[str, object]:
    return {
        "summary": "Test a new optimizer mechanism without changing scale",
        "contribution_level": "method_discovery",
        "target_component": "optimizer",
        "mechanism": "Decouple update direction from adaptive magnitude",
        "metric": {
            "name": "accuracy",
            "direction": "maximize",
            "minimum_effect": 0.01,
            "theoretical_bound": 1.0,
        },
        "edit_scope": {
            "allowed_paths": ["src/method.py"],
            "protected_paths": ["src/model.py", "src/data.py", "src/evaluator.py"],
        },
        "fixed_variables": {
            "model": "tiny-transformer-v1",
            "training_steps": 1000,
            "data_version": "dataset-v1",
        },
        "resource_limits": {"gpu_hours": 10, "parameters": 1000000},
        "runner": {"entrypoint": "python evaluate.py", "seeds": [1, 2, 3]},
        "baselines": [
            {
                "id": "adamw",
                "method": "AdamW",
                "source": "doi:10.48550/arXiv.1711.05101",
            },
            {"id": "lion", "method": "Lion", "source": "doi:10.48550/arXiv.2302.06675"},
            {
                "id": "schedule_free",
                "method": "Schedule-Free AdamW",
                "source": "commit:baseline-schedule-free",
            },
        ],
        "conditions": [
            {
                "id": "dev-small",
                "role": "development",
                "visibility": "visible",
                "dataset": "dataset-a",
                "scale": "proxy",
                "proxy_for": "scale-large",
            },
            {
                "id": "transfer-b",
                "role": "transfer",
                "visibility": "sealed",
                "dataset": "dataset-b",
                "scale": "proxy",
            },
            {
                "id": "scale-large",
                "role": "scale",
                "visibility": "sealed",
                "dataset": "dataset-a",
                "scale": "target",
            },
        ],
    }


def _results(
    runner_hash: str, *, changed_path: str = "src/method.py"
) -> dict[str, object]:
    return {
        "candidate": {"id": "candidate-a", "changed_paths": [changed_path]},
        "runner_hash": runner_hash,
        "fixed_variables": {
            "model": "tiny-transformer-v1",
            "training_steps": 1000,
            "data_version": "dataset-v1",
        },
        "resource_usage": {"gpu_hours": 8, "parameters": 1000000},
        "condition_results": [
            {
                "condition_id": "dev-small",
                "candidate": 0.80,
                "baselines": {"adamw": 0.72, "lion": 0.74, "schedule_free": 0.75},
            },
            {
                "condition_id": "transfer-b",
                "candidate": 0.77,
                "baselines": {"adamw": 0.69, "lion": 0.71, "schedule_free": 0.72},
            },
            {
                "condition_id": "scale-large",
                "candidate": 0.83,
                "baselines": {"adamw": 0.76, "lion": 0.78, "schedule_free": 0.79},
            },
        ],
    }


@unittest.skipUnless(shutil.which("git"), "Git is required for discovery tests")
class ResearchDiscoveryTests(unittest.TestCase):
    def _repository(self, root: Path) -> ResearchRepository:
        return ResearchRepository.init(
            root,
            question="Does the mechanism transfer?",
            git_user_name="Research Test",
            git_user_email="research@example.invalid",
        )

    def test_contract_locks_scope_budget_blinding_and_commitments(self) -> None:
        built = build_discovery_contract("rso-hypothesis", _spec())

        self.assertEqual(built["contract"]["contribution_level"], "method_discovery")
        self.assertEqual(len(built["contract"]["baselines"]), 3)
        self.assertEqual(
            built["blinding"]["sealed_condition_ids"],
            ["scale-large", "transfer-b"],
        )
        self.assertEqual(
            research_payload_issues("experiment_design", built["contract"]), []
        )
        tampered = dict(built["contract"])
        tampered["mechanism"] = "post-hoc changed mechanism"
        self.assertIn(
            "method discovery design_hash mismatch",
            research_payload_issues("experiment_design", tampered),
        )

    def test_assessment_rejects_resource_or_edit_scope_shortcuts(self) -> None:
        built = build_discovery_contract("rso-hypothesis", _spec())
        results = _results(
            str(built["contract"]["runner_hash"]), changed_path="src/model.py"
        )
        assessment = assess_generalization(
            built["contract"],
            results,
            locked_resource_limits=built["budget"]["limits"],
        )

        self.assertEqual(assessment["verdict"], "invalid_protocol_execution")
        failed = {row["code"] for row in assessment["checks"] if not row["passed"]}
        self.assertIn("target_variable_isolated", failed)

    def test_visible_gain_without_transfer_is_not_method_discovery(self) -> None:
        built = build_discovery_contract("rso-hypothesis", _spec())
        results = _results(str(built["contract"]["runner_hash"]))
        condition_rows = results["condition_results"]
        assert isinstance(condition_rows, list)
        for row in condition_rows:
            if isinstance(row, dict) and row["condition_id"] != "dev-small":
                row["candidate"] = 0.60

        assessment = assess_generalization(
            built["contract"],
            results,
            locked_resource_limits=built["budget"]["limits"],
        )

        self.assertEqual(assessment["verdict"], "engineering_gain_only")
        self.assertFalse(assessment["method_discovery_supported"])

    def test_supported_assessment_can_anchor_method_discovery_claim(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repository = self._repository(Path(td) / "research")
            hypothesis = repository.record(
                "hypothesis", {"statement": "A decoupled update transfers"}
            )
            context = record_research_context_snapshot(
                repository,
                target_ids=[hypothesis.object_id],
                decision_kind="method_discovery_design",
                rationale=["Mechanism addresses the recorded failure mode"],
            )
            contract_result = save_discovery_contract(
                str(repository.path),
                hypothesis_id=hypothesis.object_id,
                spec=_spec(),
                context_id=context.object_id,
                commit=False,
            )
            contract_id = contract_result["object"].object_id
            contract_object = repository.get(contract_id)
            contract = contract_object["payload"]
            self.assertEqual(contract["context_id"], context.object_id)
            self.assertTrue(
                any(
                    row.get("type") == "uses_context"
                    and row.get("target") == context.object_id
                    for row in contract_object["relations"]
                )
            )
            plan = repository.record(
                "research_plan",
                {"summary": "Run every locked discovery condition"},
                relations=[{"type": "depends_on", "target": hypothesis.object_id}],
            )
            attempt = repository.record(
                "experiment_attempt",
                {"status": "completed", "study_phase": "exploratory"},
                state="completed",
                relations=[
                    {"type": "depends_on", "target": plan.object_id, "role": "plan"}
                ],
            )
            evidence = repository.record(
                "evidence",
                {
                    "result": "all locked conditions completed",
                    "measurement_hash": "sha256:" + "a" * 64,
                },
                state="completed",
                relations=[{"type": "derived_from", "target": attempt.object_id}],
            )
            assessment_result = save_generalization_assessment(
                str(repository.path),
                contract_id=contract_id,
                results=_results(str(contract["runner_hash"])),
                evidence_ids=[evidence.object_id],
                commit=False,
            )
            self.assertEqual(
                assessment_result["assessment"]["verdict"],
                "method_discovery_supported",
            )
            claim_result = save_claim(
                str(repository.path),
                statement="The mechanism improves across conditions.",
                evidence_ids=[assessment_result["object"].object_id],
                contribution_level="method_discovery",
                commit=False,
            )
            repository.commit(stage="claim", subject="record discovery proof")

            audit = repository.audit(level="trace")

            self.assertTrue(audit["complete"], audit["blockers"])
            self.assertEqual(claim_result["object"].state, "draft")

    def test_cli_plan_accepts_one_json_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repository = self._repository(root / "research")
            hypothesis = repository.record("hypothesis", {"statement": "H1"})
            spec_path = root / "discovery.json"
            spec_path.write_text(json.dumps(_spec()), encoding="utf-8")
            output = io.StringIO()

            with redirect_stdout(output):
                status = research_main(
                    [
                        "discovery",
                        "plan",
                        hypothesis.object_id,
                        str(spec_path),
                        "--repo",
                        str(repository.path),
                        "--no-commit",
                        "--json",
                    ]
                )

            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["object"]["kind"], "experiment_design")
            self.assertEqual(len(payload["related_objects"]), 2)

    def test_cli_template_gives_beginner_a_complete_starting_contract(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            destination = Path(td) / "discovery.json"
            output = io.StringIO()

            with redirect_stdout(output):
                status = research_main(
                    [
                        "discovery",
                        "template",
                        "--output",
                        str(destination),
                        "--json",
                    ]
                )

            self.assertEqual(status, 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(
                payload["template"]["contribution_level"], "method_discovery"
            )
            self.assertTrue(destination.is_file())
            self.assertEqual(len(payload["template"]["conditions"]), 3)


if __name__ == "__main__":
    unittest.main()
