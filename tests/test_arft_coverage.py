from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.arft_coverage import (
    ARFT_PATTERN_CATALOG,
    ARFT_SCHEMA,
    ARFT_STAGES,
    build_arft_coverage,
    save_arft_coverage,
)
from ai_scientist.utils.experiment_registry import save_experiment_registry
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    load_contract_artifact,
    save_contract_artifact,
)
import xscientist
from jsonschema import validate

from ai_scientist.protocol.schemas import load_schema


class ARFTCoverageTests(unittest.TestCase):
    def test_public_sdk_exports_are_lazy_and_callable(self) -> None:
        self.assertTrue(callable(xscientist.build_arft_coverage))
        self.assertTrue(callable(xscientist.save_arft_coverage))
        self.assertTrue(callable(xscientist.build_process_summary))

    def test_empty_workspace_is_explicitly_unassessed_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "empty"
            root.mkdir()
            report = build_arft_coverage(root)

            self.assertEqual(report["schema"], ARFT_SCHEMA)
            self.assertFalse(report["quality_claim_allowed"])
            self.assertFalse(report["benchmark_compatible"])
            self.assertEqual(report["summary"]["pattern_count"], 45)
            self.assertEqual(len(report["stages"]), len(ARFT_STAGES))
            self.assertEqual(
                report["summary"]["unassessed_pattern_count"],
                len(ARFT_PATTERN_CATALOG),
            )
            # Computing a report must not bootstrap a pipeline manifest.
            self.assertFalse((root / "pipeline_manifest.json").exists())

    def test_contract_signals_cover_multiple_arft_stages_without_calling_provider(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            initialize_pipeline_contracts(
                root,
                pipeline_goal="test whether traceable research improves reliability",
            )
            save_contract_artifact(
                root,
                "idea_cards",
                [
                    {
                        "idea_id": "i1",
                        "core_hypothesis": "Traceability improves reliability.",
                        "novelty_claim": "Explicit provenance binding.",
                        "related_work_notes": "Compared with prior workflows.",
                        "alternative_framing": "Test failure recovery instead.",
                        "failure_criteria": ["No improvement on held-out data."],
                        "candidate_metrics": ["reliability"],
                        "literature_queries": ["research provenance"],
                    }
                ],
                producer="test",
            )
            save_contract_artifact(
                root,
                "research_plan",
                {
                    "budget": {
                        "max_steps": 4,
                        "max_wallclock_minutes": 5,
                        "max_retry_per_task": 1,
                    },
                    "tasks": [
                        {
                            "task_id": "t1",
                            "hypothesis_id": "i1",
                            "claim_targets": ["c1"],
                            "success_criterion": "Reliability improves.",
                            "method": "held-out comparison",
                            "candidate_baselines": ["base"],
                            "leakage_check": "sealed split",
                        }
                    ],
                    "socratic_challenge": {
                        "rival_hypotheses": ["No effect"],
                    },
                },
                producer="test",
            )
            save_contract_artifact(
                root,
                "claim_evidence_graph",
                {
                    "nodes": [
                        {"id": "h1", "type": "hypothesis"},
                        {"id": "c1", "type": "claim", "source": "doi:demo"},
                    ],
                    "edges": [
                        {
                            "source": "h1",
                            "target": "c1",
                            "supports": "doi:demo",
                        }
                    ],
                },
                producer="test",
            )
            save_experiment_registry(
                root,
                [
                    {
                        "record_id": "r1",
                        "task_id": "t1",
                        "status": "completed",
                        "method": "held-out comparison",
                        "code_ref": "run.py",
                        "seed": 7,
                        "baseline_ref": "base",
                        "data_validation": "passed",
                        "leakage_check": "passed",
                        "result_summary": {
                            "metric": "reliability",
                            "value": 0.9,
                            "confidence_interval": [0.8, 0.95],
                        },
                        "provenance": {"command": "python run.py"},
                    }
                ],
            )
            save_contract_artifact(
                root,
                "manuscript_state",
                {
                    "claim_bindings": {"c1": "r1"},
                    "references": ["doi:demo"],
                    "limitations": ["Only one dataset."],
                    "negative_results": ["Ablation was neutral."],
                },
                producer="test",
            )
            save_contract_artifact(
                root,
                "review_state",
                {
                    "rounds": [
                        {"role": "adversarial_skeptic", "evidence_anchors": ["r1"]}
                    ],
                    "repair_actions": ["re-run held-out check"],
                    "verification_checks": ["claim bound to result"],
                },
                producer="test",
            )

            report = build_arft_coverage(root)
            by_id = {item["id"]: item for item in report["patterns"]}
            self.assertEqual(by_id["A.2"]["status"], "covered")
            self.assertEqual(by_id["C.3"]["status"], "covered")
            self.assertEqual(by_id["E.1"]["status"], "covered")
            self.assertEqual(by_id["F.3"]["status"], "covered")
            self.assertEqual(by_id["X.3"]["status"], "covered")
            self.assertGreater(report["summary"]["coverage_score"], 0.0)
            self.assertIn("experiment_registry", by_id["C.4"]["evidence_channels"])
            validate(report, load_schema("arft_coverage"))

    def test_save_arft_coverage_is_a_round_trippable_pipeline_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            initialize_pipeline_contracts(root)
            path = save_arft_coverage(root)
            self.assertEqual(Path(path).name, "arft_coverage.json")
            loaded = load_contract_artifact(root, "arft_coverage", default={})
            self.assertEqual(loaded["schema"], ARFT_SCHEMA)
            manifest = json.loads(
                (root / "pipeline_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["artifacts"]["arft_coverage"]["status"], "ready")

    def test_malformed_contract_is_reported_without_leaking_file_contents(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            root.mkdir()
            (root / "idea_cards.json").write_text(
                '{"private": "do-not-leak",', encoding="utf-8"
            )
            (root / "pipeline_manifest.json").write_text(
                '{"private": "manifest-do-not-leak",', encoding="utf-8"
            )
            report = build_arft_coverage(root)

        self.assertEqual(
            {item["artifact"] for item in report["input_errors"]},
            {"idea_cards", "pipeline_manifest"},
        )
        self.assertTrue(
            all(item["error"] == "invalid_json" for item in report["input_errors"])
        )
        self.assertNotIn("do-not-leak", json.dumps(report))
        self.assertNotIn("manifest-do-not-leak", json.dumps(report))
        validate(report, load_schema("arft_coverage"))

    def test_artifact_channel_exists_does_not_follow_symlink_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            outside = Path(td) / "outside.json"
            root.mkdir()
            outside.write_text('{"private": "outside"}', encoding="utf-8")
            (root / "idea_cards.json").symlink_to(outside)

            report = build_arft_coverage(root)

        self.assertFalse(report["artifact_channels"]["idea_cards"]["exists"])
        self.assertIn(
            {"artifact": "idea_cards", "error": "symlink_boundary"},
            report["input_errors"],
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
