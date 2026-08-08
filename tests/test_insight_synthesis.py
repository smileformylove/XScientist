from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import validate

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.insight_synthesis import synthesize_project_insights
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    save_contract_artifact,
)


class InsightSynthesisTests(unittest.TestCase):
    def _project(self, root: Path) -> tuple[Path, list[dict]]:
        initialize_pipeline_contracts(root, project_name="insight-demo")
        save_contract_artifact(
            root,
            "idea_cards",
            [
                {
                    "name": "mechanism_probe",
                    "title": "Mechanism probe",
                    "core_hypothesis": "The intervention improves the metric.",
                    "mechanism": "It reduces noisy retrieval.",
                    "falsifiers": ["No paired improvement."],
                }
            ],
            producer="test",
        )
        save_contract_artifact(
            root,
            "research_plan",
            {
                "socratic_challenge": {
                    "rival_hypotheses": [
                        {
                            "rival_id": "rival_null",
                            "class": "null_effect",
                            "statement": "The delta is run-to-run variation.",
                            "discriminating_prediction": "It vanishes across seeds.",
                        }
                    ]
                }
            },
            producer="test",
        )
        exp = root / "02_experiments" / "run"
        exp.mkdir(parents=True)
        (exp / "experiment_report.json").write_text(
            json.dumps(
                {
                    "stages": [
                        {
                            "stage_dir": "stage_0_baseline",
                            "best": {
                                "metric_name": "accuracy",
                                "metric_mean": 0.5,
                                "metric_objective": 0.5,
                                "dataset_names": ["demo"],
                            },
                            "node_counts": {"total": 2, "good": 2, "buggy": 0},
                        },
                        {
                            "stage_dir": "stage_1_search",
                            "best": {
                                "metric_name": "accuracy",
                                "metric_mean": 0.6,
                                "metric_objective": 0.6,
                                "dataset_names": ["demo"],
                            },
                            "delta_objective_vs_prev_stage": 0.1,
                            "node_counts": {"total": 3, "good": 3, "buggy": 0},
                        },
                    ],
                    "warnings": [],
                }
            ),
            encoding="utf-8",
        )
        return exp, [{"idea_idx": 0, "status": "success", "exp_dir": str(exp)}]

    def test_deterministic_report_is_schema_valid_and_never_claims_verification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _exp, results = self._project(root)
            report = synthesize_project_insights(root, results)

            validate(report, load_schema("insight_report"))
            unsigned = dict(report)
            report_hash = unsigned.pop("report_hash")
            self.assertEqual(report_hash, canonical_content_hash(unsigned))
            self.assertFalse(report["independent_verification"])
            self.assertEqual(
                report["insights"][0]["epistemic_status"],
                "machine_synthesized_unverified",
            )
            self.assertIn("metric:0:0", report["insights"][0]["evidence_refs"])
            self.assertTrue((root / "04_logs" / "insight_report.md").is_file())

    def test_invalid_model_certainty_falls_back_to_grounded_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _exp, results = self._project(root)

            def response_fn(**_kwargs):
                return (
                    '```json\n{"insights":[{"idea_idx":0,"title":"Bad",'
                    '"claim":"This is conclusively proven","kind":"mechanism",'
                    '"confidence":"high","evidence_refs":["invented:1"]}]}\n```',
                    [],
                )

            report = synthesize_project_insights(
                root,
                results,
                model="demo",
                use_llm=True,
                client_factory=lambda model: (object(), model),
                response_fn=response_fn,
            )

            self.assertEqual(report["synthesis_mode"], "deterministic_fallback")
            self.assertTrue(report["insights"])
            self.assertEqual(report["insights"][0]["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
