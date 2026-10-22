from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from copy import deepcopy
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_artifacts import (
    build_evolution_candidate_from_sources,
)
from ai_scientist.utils.evolution_runtime import (
    BENCHMARK_SUITE_SCHEMA,
    EvolutionRuntimeError,
)
from ai_scientist.utils.harness_refinement import SPEC_SCHEMA, run_harness_refinement
from ai_scientist.utils.science_constitution import build_science_constitution
from xscientist.evolution_cli import main

EVALUATOR = """import json
import os
import runpy
from pathlib import Path

root = Path(os.environ["XSCIENTIST_ARTIFACT_ROOT"])
parser = runpy.run_path(str(root / "parse.py"))
estimator = runpy.run_path(str(root / "estimate.py"))
holdout = os.environ["XSCIENTIST_TASK_ID"] == "holdout"
inputs = ["1200", "2400"] if holdout else ["1,000", "2,000"]
parsed = []
events = []
for value in inputs:
    try:
        parsed.append(parser["parse"](value))
    except ValueError:
        pass
for module, name in ((parser, "parse"), (estimator, "estimate")):
    if module.get("CHANGE_ID") and not os.environ.get("OMIT_EVENTS"):
        events.append({"event_id": "call-" + name, "change_id": module["CHANGE_ID"],
                       "kind": "activated", "detail": "Evaluator invoked " + name})
estimate = estimator["estimate"]([2, 4, 6] if holdout else [1, 3, 5])
score = (len(parsed) / len(inputs) + float(estimate == (4 if holdout else 3))) / 2
if os.environ.get("NOISY_BASE") and not events:
    score = float(os.environ["XSCIENTIST_REFINEMENT_REPEAT"])
metrics = {"objective_score": score, "reproducibility_rate": 1.0,
           "false_discovery_rate": 0.0, "cost_per_task": 0.01, "latency_seconds": 0.001}
if os.environ.get("MISSING_METRIC"):
    del metrics["reproducibility_rate"]
if os.environ.get("BAD_EVENTS"):
    events = [{"event_id": "incomplete"}]
if os.environ.get("COUNTEREXAMPLE"):
    events.append({"event_id": "counterexample", "change_id": "inv", "kind": "counterexample",
                   "detail": "Observed failure on a valid input"})
print(json.dumps({"metrics": metrics, "safety_pass": not bool(os.environ.get("UNSAFE")),
                  "integrity_pass": True, "reproducibility_pass": True,
                  "behavior_evidence": events}))
"""


class HarnessRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = self.root / "cas"
        self.constitution = build_science_constitution(project_name="refinement-test")
        self.base_files = {
            "parse.py": "CHANGE_ID = None\ndef parse(value):\n    return int(value)\n",
            "estimate.py": "CHANGE_ID = None\ndef estimate(values):\n    return sum(values)\n",
        }
        self._source("base", self.base_files)
        self.evaluator = self.root / "evaluate.py"
        self.evaluator.write_text(EVALUATOR, encoding="utf-8")
        self.spec = {
            "schema_version": SPEC_SCHEMA,
            "proposals": [
                self._proposal(
                    "inv",
                    "invariant",
                    {
                        "parse.py": "CHANGE_ID = 'inv'\ndef parse(value):\n    return int(value.replace(',', ''))\n"
                    },
                ),
                self._proposal(
                    "dom",
                    "domain",
                    {
                        "estimate.py": "CHANGE_ID = 'dom'\ndef estimate(values):\n    return sum(values) / len(values)\n"
                    },
                ),
            ],
            "development_suite": self._suite("development", "public"),
            "holdout_suite": self._suite("holdout", "hidden"),
        }

    def _source(self, name: str, files: dict[str, str]) -> Path:
        root = self.root / name
        for path, text in files.items():
            target = root / "failure_recovery" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return root

    def _proposal(self, pid: str, dimension: str, edits: dict[str, str]) -> dict:
        result = build_evolution_candidate_from_sources(
            constitution=self.constitution,
            base_root=self.root / "base",
            candidate_root=self._source(pid, {**self.base_files, **edits}),
            store_root=self.store,
            candidate_id=pid,
            component_type="failure_recovery",
            base_version="1.0",
            candidate_version="1.1-" + pid,
            proposed_by="agent:test",
            change_summary="Test one observable mechanism.",
            change_scope=["failure_recovery/" + path for path in edits],
            applicability_domains=["numeric-experiments"],
            failure_taxonomy_refs=["failure:" + pid],
            ablation_dimensions=[pid],
        )
        return {
            "proposal_id": pid,
            "dimension": dimension,
            "candidate": result["candidate"],
            "cue": "Numeric experimental inputs",
            "hypothesis": "The changed procedure improves valid results.",
            "falsifier": "An observed regression or absent activation rejects the procedure.",
        }

    def _suite(self, task: str, split: str) -> dict:
        return {
            "schema_version": BENCHMARK_SUITE_SCHEMA,
            "benchmark_id": task,
            "producer_stack_id": "agent:test",
            "producer_stack_hash": canonical_content_hash("producer"),
            "evaluator_stack_id": "service:test",
            "evaluator_stack_hash": canonical_content_hash("evaluator"),
            "evaluator_id": "service:test",
            "tasks": [
                {
                    "task_id": task,
                    "task_hash": canonical_content_hash(task),
                    "split": split,
                    "evaluation_layer": "public" if split == "public" else "sealed",
                    "frozen_before_candidate": True,
                    "command": [sys.executable, str(self.evaluator)],
                }
            ],
        }

    def _environment(self, **environment: str) -> None:
        for suite in ("development_suite", "holdout_suite"):
            self.spec[suite]["tasks"][0]["environment"] = environment

    def _run(self, execute: bool = True) -> dict:
        result = run_harness_refinement(
            self.spec,
            constitution=self.constitution,
            artifact_store=self.store,
            allow_execution=execute,
        )
        self.assertEqual(
            result["report_hash"],
            canonical_content_hash(
                {key: value for key, value in result.items() if key != "report_hash"}
            ),
        )
        self.assertFalse(result["production_promotion_allowed"])
        self.assertFalse(result["persistent_learning_allowed"])
        return result

    def test_plan_never_executes_or_composes(self) -> None:
        with patch(
            "ai_scientist.utils.harness_refinement.run_shadow_benchmark"
        ) as execute:
            result = self._run(False)
        execute.assert_not_called()
        self.assertEqual(result["status"], "planned")
        self.assertEqual(result["execution_count"], 0)
        self.assertEqual(result["planned_max_executions"], 16)

    def test_real_probes_compose_and_select_only_after_holdout(self) -> None:
        original = deepcopy(self.spec)
        result = self._run()
        self.assertEqual(self.spec, original)
        self.assertEqual(result["decision"], "shadow_candidate_selected")
        self.assertEqual(result["execution_count"], 16)
        self.assertAlmostEqual(result["observed_cost"], 0.16)
        self.assertEqual(result["accounting"]["baseline_executions"], 8)
        self.assertEqual(result["accounting"]["candidate_executions"], 8)
        self.assertGreater(result["accounting"]["process_seconds"], 0)
        self.assertFalse(result["accounting"]["proposal_generation_cost_observed"])
        self.assertEqual(result["composition"]["status"], "probed")
        self.assertEqual(
            result["selected_candidate"]["candidate_hash"], result["development_winner"]
        )
        self.assertTrue(result["holdout"]["eligible"])
        self.assertEqual(result["trials"][-1]["mean_improvement"], 1.0)
        for experience in result["experience"]:
            self.assertEqual(experience["status"], "quarantined")
            self.assertTrue(experience["holdout_report_hashes"])
            self.assertTrue(experience["evidence"][0]["run_hash"].startswith("sha256:"))
        for trial in [*result["trials"], result["holdout"]]:
            for run in trial["runs"]:
                for receipt in run["run_receipts"]:
                    self.assertEqual(
                        receipt["run_hash"],
                        canonical_content_hash(
                            {
                                key: value
                                for key, value in receipt.items()
                                if key != "run_hash"
                            }
                        ),
                    )

    def test_absent_activation_abstains_even_if_score_improves(self) -> None:
        self._environment(OMIT_EVENTS="1")
        result = self._run()
        self.assertEqual(result["decision"], "keep_base")
        self.assertIsNone(result["holdout"])
        self.assertIn("activation_not_observed", result["trials"][0]["reasons"])

    def test_counterexample_blocks_mechanism(self) -> None:
        self._environment(COUNTEREXAMPLE="1")
        result = self._run()
        self.assertIn("mechanism_counterexample", result["trials"][0]["reasons"])
        self.assertEqual(result["composition"]["status"], "not_attempted")

    def test_no_holdout_leaves_development_winner_advisory(self) -> None:
        del self.spec["holdout_suite"]
        result = self._run()
        self.assertTrue(result["development_winner"])
        self.assertIsNone(result["selected_candidate"])
        self.assertEqual(result["decision"], "keep_base")

    def test_holdout_failure_does_not_choose_a_runner_up(self) -> None:
        self.spec["proposals"][1] = self._proposal(
            "bad",
            "domain",
            {"estimate.py": "CHANGE_ID = 'bad'\ndef estimate(values):\n    return 3\n"},
        )
        result = self._run()
        self.assertEqual(result["decision"], "keep_base")
        self.assertIsNone(result["selected_candidate"])
        self.assertTrue(result["development_winner"])
        self.assertFalse(result["holdout"]["eligible"])
        self.assertEqual(len(result["holdout"]["runs"]), 2)
        self.assertIn(
            "holdout:does_not_beat_repeated_baseline",
            result["experience"][1]["counterexamples"],
        )

    def test_repeated_baseline_controls_lucky_improvement(self) -> None:
        self._environment(NOISY_BASE="1")
        result = self._run()
        self.assertEqual(result["decision"], "keep_base")
        self.assertIn("does_not_beat_repeated_baseline", result["trials"][0]["reasons"])

    def test_scientific_failure_halts_and_preserves_receipts(self) -> None:
        self._environment(UNSAFE="1")
        result = self._run()
        self.assertEqual(result["status"], "halted")
        self.assertEqual(result["stop_reason"], "scientific_gate_failed")
        self.assertEqual(result["execution_count"], 2)
        self.assertEqual(result["decision"], "keep_base")
        self.assertTrue(result["trials"][0]["runs"])

    def test_execution_cap_is_checked_before_any_execution(self) -> None:
        self.spec["policy"] = {"max_executions": 2}
        with patch(
            "ai_scientist.utils.harness_refinement.run_shadow_benchmark"
        ) as execute:
            with self.assertRaisesRegex(EvolutionRuntimeError, "max_executions"):
                self._run()
        execute.assert_not_called()

    def test_cost_limit_stops_at_pair_boundary(self) -> None:
        self.spec["policy"] = {"max_cost": 0.02}
        result = self._run()
        self.assertEqual(result["execution_count"], 2)
        self.assertEqual(result["stop_reason"], "cost_budget_exhausted")

    def test_missing_quality_metric_never_selects(self) -> None:
        self._environment(MISSING_METRIC="1")
        result = self._run()
        self.assertEqual(result["decision"], "keep_base")
        self.assertIn(
            "required_metric_missing_or_invalid", result["trials"][0]["reasons"]
        )

    def test_invalid_events_keep_negative_execution_receipt(self) -> None:
        self._environment(BAD_EVENTS="1")
        result = self._run()
        self.assertEqual(result["decision"], "keep_base")
        receipt = result["trials"][0]["runs"][0]["run_receipts"][0]
        self.assertEqual(receipt["status"], "invalid_output")

    def test_conflicting_paths_are_not_composed(self) -> None:
        self.spec["proposals"][1] = self._proposal(
            "other",
            "domain",
            {
                "parse.py": "CHANGE_ID = 'other'\ndef parse(value):\n    return int(''.join(value.split(',')))\n"
            },
        )
        result = self._run()
        self.assertEqual(
            result["composition"],
            {"status": "skipped", "reason": "overlapping_changes"},
        )
        self.assertEqual(result["decision"], "keep_base")
        self.assertFalse(result["holdout"]["eligible"])

    def test_rejects_hidden_feedback_in_development(self) -> None:
        self.spec["development_suite"]["tasks"][0]["split"] = "hidden"
        with self.assertRaisesRegex(
            EvolutionRuntimeError, "development probes must be public"
        ):
            self._run(False)

    def test_rejects_overlapping_task_content_even_with_different_ids(self) -> None:
        self.spec["holdout_suite"]["tasks"][0]["task_hash"] = self.spec[
            "development_suite"
        ]["tasks"][0]["task_hash"]
        with self.assertRaisesRegex(EvolutionRuntimeError, "disjoint"):
            self._run(False)

    def test_rejects_environment_override(self) -> None:
        self._environment(XSCIENTIST_ARTIFACT_ROOT="/tmp/other")
        with self.assertRaisesRegex(EvolutionRuntimeError, "runtime-owned"):
            self._run(False)

    def test_cli_is_available_through_product_dispatch(self) -> None:
        from xscientist.cli import main as product_main

        spec = self.root / "spec.json"
        constitution = self.root / "constitution.json"
        spec.write_text(json.dumps(self.spec), encoding="utf-8")
        constitution.write_text(json.dumps(self.constitution), encoding="utf-8")
        with patch("sys.stdout", new_callable=StringIO) as stdout:
            code = product_main(
                [
                    "evolution",
                    "refine",
                    "--spec",
                    str(spec),
                    "--constitution",
                    str(constitution),
                    "--store",
                    str(self.store),
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "planned")

    def test_rejects_duplicate_candidate_and_invalid_policy(self) -> None:
        self.spec["proposals"][1]["candidate"] = self.spec["proposals"][0]["candidate"]
        with self.assertRaisesRegex(EvolutionRuntimeError, "duplicate candidate"):
            self._run(False)
        self.spec["policy"] = {"repeats": True}
        with self.assertRaisesRegex(EvolutionRuntimeError, "invalid repeats"):
            self._run(False)

    def test_cli_plans_and_protects_existing_output_before_execution(self) -> None:
        spec, constitution, output = [
            self.root / name
            for name in ("spec.json", "constitution.json", "report.json")
        ]
        spec.write_text(json.dumps(self.spec), encoding="utf-8")
        constitution.write_text(json.dumps(self.constitution), encoding="utf-8")
        argv = [
            "refine",
            "--spec",
            str(spec),
            "--constitution",
            str(constitution),
            "--store",
            str(self.store),
            "--out",
            str(output),
        ]
        self.assertEqual(main(argv), 0)
        self.assertEqual(json.loads(output.read_text())["status"], "planned")
        with patch("xscientist.evolution_cli.run_harness_refinement") as execute:
            with redirect_stderr(StringIO()):
                self.assertEqual(main([*argv, "--execute"]), 2)
        execute.assert_not_called()
        self.assertEqual(main([*argv, "--execute", "--force"]), 0)
        self.assertEqual(
            json.loads(output.read_text())["decision"], "shadow_candidate_selected"
        )


if __name__ == "__main__":
    unittest.main()
