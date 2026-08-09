from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.decision_log import (
    DecisionLogError,
    build_sample_gate_decision_options,
    build_workflow_strategy_decision_options,
    load_decision_log,
    record_decision,
    record_model_provider_decisions,
)
from ai_scientist.utils.pipeline_contracts import (
    artifact_path,
    initialize_pipeline_contracts,
    load_pipeline_manifest,
)


class DecisionLogTests(unittest.TestCase):
    def test_record_decision_should_persist_auditable_options(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)

            entry = record_decision(
                project_root,
                category="runtime_mode",
                selected="simulate",
                options_considered=["simulate", "live"],
                rejected_because={"live": "no user-approved live budget"},
                producer="test_decision_log",
                metadata={"source": "unit"},
            )

            self.assertEqual(entry["selected"], "simulate")
            self.assertTrue(entry["decision_input_hash"].startswith("sha256:"))
            self.assertTrue(entry["decision_hash"].startswith("sha256:"))
            self.assertEqual(
                entry["options_considered"][1]["rejected_because"],
                "no user-approved live budget",
            )
            self.assertEqual(
                load_decision_log(project_root)[0]["category"], "runtime_mode"
            )
            manifest = load_pipeline_manifest(project_root)
            self.assertEqual(manifest["artifacts"]["decision_log"]["status"], "ready")

    def test_record_decision_should_reject_single_option_without_rejected_reason(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)

            with self.assertRaises(DecisionLogError):
                record_decision(
                    project_root,
                    category="runtime_mode",
                    selected="simulate",
                    options_considered=["simulate"],
                    producer="test_decision_log",
                )

            with self.assertRaises(DecisionLogError):
                record_decision(
                    project_root,
                    category="runtime_mode",
                    selected="simulate",
                    options_considered=["simulate"],
                    rejected_because={"live": "not approved"},
                    producer="test_decision_log",
                )

    def test_record_decision_should_reject_unexplained_non_selected_options(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)

            with self.assertRaises(DecisionLogError):
                record_decision(
                    project_root,
                    category="provider",
                    selected="provider_a",
                    options_considered=["provider_a", "provider_b"],
                    producer="test_decision_log",
                )

    def test_record_decision_should_reject_selected_value_outside_options(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)

            with self.assertRaises(DecisionLogError):
                record_decision(
                    project_root,
                    category="provider",
                    selected="provider_c",
                    options_considered=["provider_a", "provider_b"],
                    rejected_because={
                        "provider_a": "not selected",
                        "provider_b": "not selected",
                    },
                    producer="test_decision_log",
                )

    def test_loading_v2_decision_log_rejects_tampered_context_binding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)
            record_decision(
                project_root,
                category="promotion",
                selected="hold",
                options_considered=["hold", "promote"],
                rejected_because={"promote": "verification is incomplete"},
                producer="test_decision_log",
                context_refs=["sha256:" + "a" * 64],
                memory_refs=["sha256:" + "b" * 64],
                evidence_refs=["sha256:" + "c" * 64],
            )
            path = artifact_path(project_root, "decision_log")
            entry = json.loads(path.read_text(encoding="utf-8"))
            entry["context_refs"] = ["sha256:" + "d" * 64]
            path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(DecisionLogError, "input hash mismatch"):
                load_decision_log(project_root)

    def test_record_model_provider_decisions_should_log_resolved_provider_options(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "project"
            project_root.mkdir(parents=True, exist_ok=True)
            initialize_pipeline_contracts(project_root)

            entries = record_model_provider_decisions(
                project_root,
                ["zhipu/glm-4-air", "openai/gpt-4.1"],
                producer="test_decision_log",
                env={"ZHIPU_API_KEY": "zhipu-key"},
            )

            self.assertEqual(
                [entry["category"] for entry in entries],
                ["model_provider", "model_provider"],
            )
            self.assertEqual(entries[0]["selected"], "zhipu")
            self.assertEqual(entries[1]["selected"], "openai")
            self.assertEqual(entries[0]["metadata"]["missing_credentials"], "")
            self.assertIn(
                "OPENAI_API_KEY", entries[1]["metadata"]["missing_credentials"]
            )
            for entry in entries:
                self.assertEqual(len(entry["options_considered"]), 2)
                rejected = [
                    option
                    for option in entry["options_considered"]
                    if not option["selected"]
                ]
                self.assertTrue(rejected)
                self.assertTrue(all(option["rejected_because"] for option in rejected))
            self.assertEqual(len(load_decision_log(project_root)), 2)

    def test_workflow_strategy_options_should_explain_actual_resolution_candidates(
        self,
    ) -> None:
        explicit = build_workflow_strategy_decision_options(
            selected="classic_pipeline",
            requested_workflow_mode="classic_pipeline",
        )
        self.assertEqual(
            [item["option"] for item in explicit], ["adaptive", "classic_pipeline"]
        )
        self.assertTrue(explicit[1]["selected"])
        self.assertIn("bypassed", explicit[0]["rejected_because"])

        adaptive = build_workflow_strategy_decision_options(
            selected="writing_studio",
            requested_workflow_mode="adaptive",
            high_quality_mode=True,
            target_venue="neurips",
        )
        self.assertIn("adaptive", [item["option"] for item in adaptive])
        self.assertIn("writing_studio", [item["option"] for item in adaptive])
        self.assertLess(len(adaptive), 7)
        self.assertTrue(
            all(
                item["selected"]
                or item["rejected_because"]
                != "not selected by workflow_mode/submission/high_quality/venue resolution"
                for item in adaptive
            )
        )

    def test_sample_gate_options_should_explain_rejected_option_directly(self) -> None:
        passed = build_sample_gate_decision_options(True)
        block_option = next(
            item for item in passed if item["option"] == "block_full_generation"
        )
        self.assertFalse(block_option["selected"])
        self.assertEqual(
            block_option["rejected_because"],
            "sample gate passed, no blocking reason remains",
        )

        failed = build_sample_gate_decision_options(False)
        continue_option = next(
            item for item in failed if item["option"] == "continue_full_generation"
        )
        self.assertFalse(continue_option["selected"])
        self.assertEqual(
            continue_option["rejected_because"],
            "sample gate did not pass, so full generation remains blocked",
        )


if __name__ == "__main__":
    unittest.main()
