from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.apps.project import (
    _build_experiment_registry_rows as build_project_registry_rows,
)
from ai_scientist.utils.evidence_snapshot import canonical_hash
from ai_scientist.utils.experiment_registry import build_experiment_record
from ai_scientist.utils.high_quality_pipeline import assess_experiment_rigor
from ai_scientist.utils.research_integrity import (
    _protocol_fidelity_hash,
    build_preregistration,
    validate_preregistration,
)
from ai_scientist.utils.research_planning import (
    build_claim_evidence_graph,
    build_idea_cards,
    build_research_plan,
)


class StructuredEvidencePortfolioTests(unittest.TestCase):
    def _idea_card(self, *, target_venue: str = "icml") -> dict:
        return build_idea_cards(
            [
                {
                    "Name": "Structured evidence",
                    "Short Hypothesis": "A proposed method improves accuracy.",
                    "Experiments": ["Compare the full method with a strong baseline."],
                }
            ],
            target_venue=target_venue,
        )[0]

    def _record(
        self,
        task_id: str,
        *,
        evidence_role: str,
        metric_mean: float,
        paired_control_task_id: str | None = None,
        intervention_variant: str | None = None,
        stress_condition: str | None = None,
    ) -> dict:
        primary_config = {"planning_module": True, "label_noise": 0.0}
        config = dict(primary_config)
        transformation_manifest = None
        if evidence_role == "ablation":
            config["planning_module"] = False
            transformation_manifest = {
                "evidence_role": "ablation",
                "paired_control_task_id": paired_control_task_id,
                "intervention_variant": intervention_variant,
                "stress_condition": None,
                "base_configuration_hash": canonical_hash(primary_config),
                "resulting_configuration_hash": canonical_hash(config),
                "changed_factors": {
                    "planning_module": {"before": True, "after": False}
                },
            }
        elif evidence_role == "robustness":
            config["label_noise"] = 0.2
            transformation_manifest = {
                "evidence_role": "robustness",
                "paired_control_task_id": paired_control_task_id,
                "intervention_variant": intervention_variant,
                "stress_condition": stress_condition,
                "base_configuration_hash": canonical_hash(primary_config),
                "resulting_configuration_hash": canonical_hash(config),
                "changed_factors": {"label_noise": {"before": 0.0, "after": 0.2}},
            }
        return build_experiment_record(
            task_id=task_id,
            dataset="demo",
            metric="accuracy",
            baseline_ref="strong-baseline",
            config=config,
            status="completed",
            result_summary={"metric_mean": metric_mean, "metric_std": 0.01},
            evidence_role=evidence_role,
            paired_control_task_id=paired_control_task_id,
            intervention_variant=intervention_variant,
            stress_condition=stress_condition,
            transformation_manifest=transformation_manifest,
            study_phase="confirmatory",
            dataset_split_hash="split-hash",
            evaluator_input_hash="input-hash",
            evaluator_result_hash="result-hash",
        )

    @staticmethod
    def _write_registry(root: Path, records: list[dict]) -> None:
        (root / "experiment_registry.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    @staticmethod
    def _write_required_plan(root: Path, records: list[dict]) -> None:
        tasks = []
        seen: set[str] = set()
        for record in records:
            task_id = str(record.get("task_id") or "")
            if not task_id or task_id in seen:
                continue
            seen.add(task_id)
            tasks.append(
                {
                    "task_id": task_id,
                    "evidence_role": record.get("evidence_role"),
                    "paired_control_task_id": record.get("paired_control_task_id"),
                    "intervention_variant": record.get("intervention_variant"),
                    "stress_condition": record.get("stress_condition"),
                }
            )
        (root / "research_plan.json").write_text(
            json.dumps(
                {
                    "evidence_portfolio": {
                        "required": True,
                        "required_roles": ["primary", "ablation", "robustness"],
                    },
                    "tasks": tasks,
                }
            ),
            encoding="utf-8",
        )

    def test_neurips_and_icml_high_quality_plans_fill_required_roles(self) -> None:
        for target_venue in ("neurips", "icml"):
            with self.subTest(target_venue=target_venue):
                idea_card = self._idea_card(target_venue=target_venue)
                plan = build_research_plan(
                    idea_card,
                    target_venue=target_venue,
                    high_quality_mode=True,
                )
                by_role = {task["evidence_role"]: task for task in plan["tasks"]}

                self.assertTrue(plan["evidence_portfolio"]["required"])
                self.assertEqual(
                    set(plan["evidence_portfolio"]["required_roles"]),
                    {"primary", "ablation", "robustness"},
                )
                self.assertEqual(
                    by_role["ablation"]["paired_control_task_id"],
                    by_role["primary"]["task_id"],
                )
                self.assertEqual(
                    by_role["robustness"]["paired_control_task_id"],
                    by_role["primary"]["task_id"],
                )
                self.assertTrue(by_role["ablation"]["intervention_variant"])
                self.assertTrue(by_role["robustness"]["stress_condition"])
                self.assertEqual(
                    by_role["ablation"]["baseline"], by_role["primary"]["baseline"]
                )
                self.assertEqual(
                    by_role["robustness"]["metric"], by_role["primary"]["metric"]
                )

                graph = build_claim_evidence_graph(idea_card, plan)
                relation_types = {edge["type"] for edge in graph["edges"]}
                self.assertIn("ablates", relation_types)
                self.assertIn("stress_tests", relation_types)

    def test_preregistration_locks_portfolio_transformation_contracts(self) -> None:
        idea_card = self._idea_card(target_venue="icml")
        plan = build_research_plan(
            idea_card,
            target_venue="icml",
            high_quality_mode=True,
        )
        preregistration = build_preregistration(idea_card, plan)
        by_role = {
            outcome["evidence_role"]: outcome for outcome in preregistration["outcomes"]
        }
        self.assertTrue(preregistration["evidence_portfolio"]["required"])
        self.assertEqual(
            by_role["ablation"]["transformation_contract"]["paired_control_task_id"],
            by_role["primary"]["task_id"],
        )
        original_protocol = _protocol_fidelity_hash(
            preregistration, by_role["ablation"]["task_id"]
        )

        by_role["ablation"]["intervention_variant"] = "renamed_only"
        tampered_protocol = _protocol_fidelity_hash(
            preregistration, by_role["ablation"]["task_id"]
        )
        validation = validate_preregistration(preregistration)

        self.assertNotEqual(original_protocol, tampered_protocol)
        self.assertIn(
            f"{by_role['ablation']['task_id']}_transformation_contract_mismatch",
            validation["errors"],
        )

    def test_normal_plan_does_not_invent_top_venue_tasks(self) -> None:
        idea_card = self._idea_card(target_venue="icml")
        plan = build_research_plan(idea_card, target_venue="icml")

        self.assertFalse(plan["evidence_portfolio"]["required"])
        self.assertEqual(len(plan["tasks"]), 1)
        self.assertIsNone(plan["tasks"][0]["evidence_role"])

    def test_registry_record_and_project_adapter_preserve_evidence_metadata(
        self,
    ) -> None:
        idea_card = self._idea_card()
        plan = build_research_plan(
            idea_card,
            target_venue="icml",
            submission_mode=True,
        )

        with tempfile.TemporaryDirectory() as td:
            rows = build_project_registry_rows(exp_dir=td, research_plan=plan)

        by_role = {row["evidence_role"]: row for row in rows}
        self.assertEqual(by_role["ablation"]["paired_control_task_id"], "task_0")
        self.assertTrue(by_role["ablation"]["intervention_variant"])
        self.assertEqual(by_role["robustness"]["paired_control_task_id"], "task_0")
        self.assertTrue(by_role["robustness"]["stress_condition"])

    def test_keywords_inside_arbitrary_json_do_not_count_as_evidence(self) -> None:
        misleading = build_experiment_record(
            task_id="task_0",
            dataset="demo",
            metric="accuracy",
            baseline_ref="strong-baseline",
            status="completed",
            config={"notes": "ablation robustness sensitivity boundary condition"},
            result_summary={"metric_mean": 0.8, "metric_std": 0.01},
            dataset_split_hash="split-hash",
            evaluator_input_hash="input-hash",
            evaluator_result_hash="result-hash",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, [misleading])
            rigor = assess_experiment_rigor(root, "ablation robustness")

        self.assertFalse(rigor["checks"]["ablation"])
        self.assertFalse(rigor["checks"]["robustness"])
        self.assertEqual(rigor["evidence_portfolio"]["valid_ablation_task_ids"], [])
        self.assertEqual(rigor["evidence_portfolio"]["valid_robustness_task_ids"], [])

    def test_structurally_paired_numeric_statistical_portfolio_passes_rigor(
        self,
    ) -> None:
        records = [
            self._record(
                "task_0",
                evidence_role="primary",
                metric_mean=0.82,
                intervention_variant="full_method",
            ),
            self._record(
                "task_1",
                evidence_role="ablation",
                metric_mean=0.76,
                paired_control_task_id="task_0",
                intervention_variant="without_planning_module",
            ),
            self._record(
                "task_2",
                evidence_role="robustness",
                metric_mean=0.79,
                paired_control_task_id="task_0",
                intervention_variant="full_method",
                stress_condition="20_percent_label_noise",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, records)
            self._write_required_plan(root, records)
            rigor = assess_experiment_rigor(root, "")

        self.assertEqual(
            rigor["checks"],
            {
                "baseline": True,
                "ablation": True,
                "statistics": True,
                "robustness": True,
                "reproducibility": True,
            },
        )
        self.assertEqual(rigor["hard_failures"], [])
        self.assertEqual(
            rigor["evidence_portfolio"]["valid_ablation_task_ids"], ["task_1"]
        )
        self.assertEqual(
            rigor["evidence_portfolio"]["valid_robustness_task_ids"], ["task_2"]
        )

    def test_broken_pairing_and_missing_structure_are_reported(self) -> None:
        records = [
            self._record(
                "task_0",
                evidence_role="primary",
                metric_mean=0.82,
                intervention_variant="full_method",
            ),
            self._record(
                "task_1",
                evidence_role="ablation",
                metric_mean=0.76,
                paired_control_task_id="missing-primary",
                intervention_variant="without_planning_module",
            ),
            self._record(
                "task_2",
                evidence_role="robustness",
                metric_mean=0.79,
                paired_control_task_id="task_0",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, records)
            rigor = assess_experiment_rigor(root, "")

        self.assertFalse(rigor["checks"]["ablation"])
        self.assertFalse(rigor["checks"]["robustness"])
        reasons = {
            item["reason"]
            for item in rigor["evidence_portfolio"]["invalid_relationships"]
        }
        self.assertIn("completed_primary_control_missing", reasons)
        self.assertIn("stress_condition_missing", reasons)

    def test_structured_labels_without_numeric_results_or_statistics_do_not_pass(
        self,
    ) -> None:
        primary = self._record(
            "task_0",
            evidence_role="primary",
            metric_mean=0.82,
            intervention_variant="full_method",
        )
        ablation = build_experiment_record(
            task_id="task_1",
            dataset="demo",
            metric="accuracy",
            baseline_ref="strong-baseline",
            config={"planning_module": False, "label_noise": 0.0},
            status="completed",
            result_summary={
                "metric_mean": "reported in manuscript",
                "metric_std": "small",
            },
            evidence_role="ablation",
            paired_control_task_id="task_0",
            intervention_variant="without_planning_module",
            transformation_manifest={
                "evidence_role": "ablation",
                "paired_control_task_id": "task_0",
                "intervention_variant": "without_planning_module",
                "stress_condition": None,
                "base_configuration_hash": primary["configuration_hash"],
                "resulting_configuration_hash": canonical_hash(
                    {"planning_module": False, "label_noise": 0.0}
                ),
                "changed_factors": {
                    "planning_module": {"before": True, "after": False}
                },
            },
            study_phase="confirmatory",
            dataset_split_hash="split-hash",
            evaluator_input_hash="input-hash",
            evaluator_result_hash="result-hash",
        )
        robustness = self._record(
            "task_2",
            evidence_role="robustness",
            metric_mean=0.79,
            paired_control_task_id="task_0",
            intervention_variant="full_method",
            stress_condition="20_percent_label_noise",
        )
        robustness["result_summary"]["metric_std"] = "small"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, [primary, ablation, robustness])
            self._write_required_plan(root, [primary, ablation, robustness])
            rigor = assess_experiment_rigor(root, "")

        self.assertFalse(rigor["checks"]["ablation"])
        self.assertFalse(rigor["checks"]["statistics"])
        reasons = {
            item["reason"]
            for item in rigor["evidence_portfolio"]["invalid_relationships"]
        }
        self.assertIn("numeric_result_missing", reasons)

    def test_exploratory_labels_cannot_satisfy_required_portfolio(self) -> None:
        records = [
            self._record(
                "task_0",
                evidence_role="primary",
                metric_mean=0.82,
                intervention_variant="full_method",
            ),
            self._record(
                "task_1",
                evidence_role="ablation",
                metric_mean=0.76,
                paired_control_task_id="task_0",
                intervention_variant="without_planning_module",
            ),
            self._record(
                "task_2",
                evidence_role="robustness",
                metric_mean=0.79,
                paired_control_task_id="task_0",
                intervention_variant="full_method",
                stress_condition="20_percent_label_noise",
            ),
        ]
        for record in records:
            record["study_phase"] = "exploratory"
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, records)
            self._write_required_plan(root, records)
            rigor = assess_experiment_rigor(root, "")

        self.assertFalse(rigor["checks"]["baseline"])
        self.assertFalse(rigor["checks"]["ablation"])
        self.assertEqual(
            rigor["evidence_portfolio"]["ignored_exploratory_record_count"], 3
        )

    def test_renamed_primary_with_unchanged_config_is_not_ablation(self) -> None:
        primary = self._record(
            "task_0",
            evidence_role="primary",
            metric_mean=0.82,
            intervention_variant="full_method",
        )
        fake_ablation = build_experiment_record(
            task_id="task_1",
            dataset="demo",
            metric="accuracy",
            baseline_ref="strong-baseline",
            config=dict(primary["config"]),
            status="completed",
            result_summary={"metric_mean": 0.82, "metric_std": 0.01},
            evidence_role="ablation",
            paired_control_task_id="task_0",
            intervention_variant="without_planning_module",
            transformation_manifest={
                "evidence_role": "ablation",
                "paired_control_task_id": "task_0",
                "intervention_variant": "without_planning_module",
                "stress_condition": None,
                "base_configuration_hash": primary["configuration_hash"],
                "resulting_configuration_hash": primary["configuration_hash"],
                "changed_factors": {
                    "planning_module": {"before": True, "after": False}
                },
            },
            study_phase="confirmatory",
            dataset_split_hash="split-hash",
            evaluator_input_hash="input-hash",
            evaluator_result_hash="result-hash",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, [primary, fake_ablation])
            self._write_required_plan(root, [primary, fake_ablation])
            rigor = assess_experiment_rigor(root, "")

        self.assertFalse(rigor["checks"]["ablation"])
        self.assertIn(
            "transformation_manifest_invalid_or_unchanged",
            {
                item["reason"]
                for item in rigor["evidence_portfolio"]["invalid_relationships"]
            },
        )

    def test_effect_size_point_estimate_is_not_uncertainty(self) -> None:
        record = build_experiment_record(
            task_id="task_0",
            dataset="demo",
            metric="accuracy",
            baseline_ref="strong-baseline",
            config={"model": "candidate"},
            status="completed",
            result_summary={
                "metric_mean": 0.82,
                "baseline_metric_mean": 0.70,
                "delta_vs_baseline": 0.12,
                "effect_size": 0.12,
            },
            study_phase="confirmatory",
            dataset_split_hash="split-hash",
            evaluator_input_hash="input-hash",
            evaluator_result_hash="result-hash",
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self._write_registry(root, [record])
            rigor = assess_experiment_rigor(root, "")

        self.assertFalse(rigor["checks"]["statistics"])


if __name__ == "__main__":
    unittest.main()
