from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.evidence_snapshot import (
    build_evidence_snapshot,
    save_evidence_snapshot,
    verify_evidence_snapshot,
)
from ai_scientist.utils.experiment_registry import (
    build_experiment_record,
    check_experiment_registry_integrity,
    save_experiment_registry,
)
from ai_scientist.utils.high_quality_pipeline import _build_submission_readiness
from ai_scientist.utils.research_integrity import (
    _canonical_hash,
    _protocol_fidelity_hash,
    build_preregistration,
    build_verification_report,
    lock_preregistration,
)


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceSnapshotIntegrityTests(unittest.TestCase):
    def test_final_manuscript_edit_invalidates_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "latex").mkdir()
            (root / "latex" / "template.tex").write_text(
                "\\section{Results}\\nAccuracy is 0.91.\\n", encoding="utf-8"
            )
            (root / "latex" / "references.bib").write_text(
                "@article{a, title={A}}\n", encoding="utf-8"
            )
            (root / "claim_evidence_graph.json").write_text(
                json.dumps({"nodes": [], "edges": []}), encoding="utf-8"
            )
            snapshot = build_evidence_snapshot(root)
            save_evidence_snapshot(root, snapshot)
            self.assertTrue(verify_evidence_snapshot(root)["ok"])

            (root / "latex" / "template.tex").write_text(
                "\\section{Results}\\nAccuracy is 0.99.\\n", encoding="utf-8"
            )
            report = verify_evidence_snapshot(root)
            self.assertFalse(report["ok"])
            self.assertIn("latex/template.tex", report["mismatches"])

    def test_registry_delete_and_reorder_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [
                build_experiment_record(
                    record_id="r1",
                    task_id="task_0",
                    dataset="d",
                    metric="accuracy",
                    baseline_ref="b",
                    status="failed",
                ),
                build_experiment_record(
                    record_id="r2",
                    task_id="task_0",
                    dataset="d",
                    metric="accuracy",
                    baseline_ref="b",
                    status="completed",
                ),
            ]
            save_experiment_registry(root, rows)
            self.assertTrue(check_experiment_registry_integrity(root)["ok"])

            registry = root / "experiment_registry.jsonl"
            registry.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[::-1]), encoding="utf-8"
            )
            report = check_experiment_registry_integrity(root)
            self.assertFalse(report["ok"])
            self.assertIn("registry_chain_tip_mismatch", report["errors"])

    def test_duplicate_seed_input_is_not_independent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prereg = lock_preregistration(
                build_preregistration(
                    {
                        "idea_id": "idea",
                        "title": "Seed independence",
                        "core_hypothesis": "The method improves accuracy.",
                    },
                    {
                        "plan_id": "plan",
                        "tasks": [
                            {
                                "task_id": "task_0",
                                "dataset": "d",
                                "metric": "accuracy",
                                "baseline": "b",
                            }
                        ],
                    },
                ),
                split_hashes={"task_0": _canonical_hash("split")},
                registered_by="planner",
            )
            input_path = root / "input.json"
            result_path = root / "result.json"
            input_path.write_text('{"seed": 1}\n', encoding="utf-8")
            result_path.write_text('{"metric_mean": 0.8}\n', encoding="utf-8")
            input_hash = _file_hash(input_path)
            result_hash = _file_hash(result_path)
            records = []
            for seed in (1, 2, 3):
                records.append(
                    {
                        "record_id": f"r{seed}",
                        "task_id": "task_0",
                        "dataset": "d",
                        "metric": "accuracy",
                        "baseline_ref": "b",
                        "status": "completed",
                        "study_phase": "confirmatory",
                        "seed": seed,
                        "producer_id": "agent",
                        "preregistration_id": prereg["preregistration_id"],
                        "protocol_fidelity_hash": _protocol_fidelity_hash(prereg, "task_0"),
                        "dataset_split_hash": prereg["data_policy"]["split_hashes"]["task_0"],
                        "evaluator_input_hash": input_hash,
                        "evaluator_result_hash": result_hash,
                        "artifacts": {"input": str(input_path), "result": str(result_path)},
                        "result_summary": {"metric_mean": 0.8},
                    }
                )
            report = build_verification_report(
                prereg,
                records,
                verifier_id="verifier",
                clean_room=True,
                verification_root=root,
            )
            self.assertIn("seed_independence", report["required_failures"])

    def test_reproduction_requirement_is_per_registered_task(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan = {
                "plan_id": "plan",
                "tasks": [
                    {"task_id": "task_0", "dataset": "d0", "metric": "accuracy", "baseline": "b0"},
                    {"task_id": "task_1", "dataset": "d1", "metric": "accuracy", "baseline": "b1"},
                ],
            }
            prereg = lock_preregistration(
                build_preregistration(
                    {"idea_id": "idea", "title": "Two tasks", "core_hypothesis": "Both improve."},
                    plan,
                ),
                split_hashes={"task_0": _canonical_hash("s0"), "task_1": _canonical_hash("s1")},
                registered_by="planner",
            )
            records = []
            for task_id, dataset, baseline in (("task_0", "d0", "b0"), ("task_1", "d1", "b1")):
                input_path = root / f"{task_id}-input.json"
                result_path = root / f"{task_id}-result.json"
                input_path.write_text(json.dumps({"task": task_id}), encoding="utf-8")
                result_path.write_text(json.dumps({"metric_mean": 0.8}), encoding="utf-8")
                input_hash = _file_hash(input_path)
                result_hash = _file_hash(result_path)
                records.append(
                    {
                        "record_id": f"primary-{task_id}",
                        "task_id": task_id,
                        "dataset": dataset,
                        "metric": "accuracy",
                        "baseline_ref": baseline,
                        "status": "completed",
                        "study_phase": "confirmatory",
                        "seed": 1,
                        "producer_id": "primary-agent",
                        "finished_at": "2026-01-01T00:00:00+00:00",
                        "preregistration_id": prereg["preregistration_id"],
                        "protocol_fidelity_hash": _protocol_fidelity_hash(prereg, task_id),
                        "dataset_split_hash": prereg["data_policy"]["split_hashes"][task_id],
                        "holdout_access": "verifier_only",
                        "artifacts": {"input": str(input_path), "result": str(result_path)},
                        "evaluator_input_hash": input_hash,
                        "evaluator_result_hash": result_hash,
                        "verification_output_hash": result_hash,
                        "verification_metric_hash": _canonical_hash({"metric_mean": 0.8}),
                        "result_summary": {"metric_mean": 0.8},
                    }
                )
                if task_id == "task_0":
                    records.append(
                        {
                            **records[-1],
                            "record_id": "reproduction-task-0",
                            "study_phase": "reproduction",
                            "independent_reproduction": True,
                            "replicates_record_id": f"primary-{task_id}",
                            "producer_id": "reproduction-agent",
                            "verifier_id": "verifier",
                            "clean_room": True,
                            "verification_recomputed": True,
                            "verification_command": "python verify_results.py",
                        }
                    )
            report = build_verification_report(
                prereg,
                records,
                verifier_id="verifier",
                clean_room=True,
                verification_root=root,
            )
            self.assertEqual(report["reproduction_task_counts"].get("task_0"), 1)
            self.assertEqual(report["reproduction_task_counts"].get("task_1"), 0)
            self.assertIn("independent_reproduction", report["required_failures"])

    def test_readiness_keeps_all_blockers(self) -> None:
        report = {
            "professional": {"overall": {"score": 5.0, "weaknesses": []}},
            "rigor": {"score": 5.0, "recommendations": []},
            "claim_support": {"score": 5.0, "recommendations": [], "unsupported_claims": []},
            "claim_alignment": {"score": 5.0},
            "numeric_coverage": {"score": 5.0},
            "breakthrough_profile": {"score": 5.0},
            "contribution_map": [{"id": "c0"}, {"id": "c1"}],
            "evidence_pack": {"num_figures": 1, "num_tables": 1, "evidence_density_score": 5.0},
            "scientific_evidence": {"status": "blocked", "hard_failures": [f"failure_{i}" for i in range(12)]},
        }
        readiness = _build_submission_readiness(
            report,
            paper_type="normal",
            target_venue="neurips",
            quality_threshold=4.0,
            rigor_threshold=3.0,
            claim_support_threshold=3.5,
        )
        self.assertEqual(readiness["blocker_count"], 12)
        self.assertEqual(len(readiness["blockers"]), 12)
        self.assertIn("scientific evidence gate: failure_11", readiness["blockers"])


if __name__ == "__main__":
    unittest.main()
