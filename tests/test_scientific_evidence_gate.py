from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from ai_scientist.utils.high_quality_pipeline import (
    _citation_integrity_report,
    _claim_graph_binding_report,
    _claim_numeric_tokens,
    _manuscript_claim_binding_report,
    _numeric_evidence_matches,
    _numeric_token_matches,
    _quality_gate_passed,
    assess_claim_support,
    assess_experiment_rigor,
    build_claim_evidence_ledger,
    build_scientific_evidence_gate,
    extract_key_results,
)
from ai_scientist.utils.evidence_snapshot import (
    build_evidence_snapshot,
    save_evidence_snapshot,
)
from ai_scientist.utils.experiment_registry import save_experiment_registry
from ai_scientist.utils.claim_registry import render_claim_prompt_snippet
from ai_scientist.utils.research_integrity import (
    _canonical_hash,
    _protocol_fidelity_hash,
    _result_artifact_manifest_present,
    _verification_output_matches,
    build_verification_report,
    build_preregistration,
    lock_preregistration,
)


def _digest(char: str) -> str:
    return "sha256:" + char * 64


def _write_hashed_json(path: Path, payload: object) -> str:
    content = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(content)
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _idea() -> dict:
    return {
        "idea_id": "idea_0",
        "title": "A falsifiable study",
        "core_hypothesis": "The intervention improves accuracy over baseline-a.",
        "failure_criteria": ["Accuracy does not improve."],
    }


def _plan() -> dict:
    return {
        "plan_id": "idea_0_plan",
        "tasks": [
            {
                "task_id": "task_0",
                "dataset": "benchmark-v1",
                "metric": "accuracy",
                "baseline": "baseline-a",
            }
        ],
    }


class ScientificEvidenceGateTests(unittest.TestCase):
    def test_numeric_claim_requires_every_reported_metric(self) -> None:
        latex = r"""
        \begin{abstract}
        We improve accuracy to 90.0\% and F1 to 99.9\% (Figure 1).
        \end{abstract}
        """
        key_results = {
            "values": ["0.9", "0.999"],
            "evidence_values": [
                {
                    "value": "0.9",
                    "metric": "accuracy",
                    "record_id": "record-1",
                    "task_id": "task-0",
                    "dataset": "benchmark",
                    "baseline": "base",
                }
            ],
        }
        report = assess_claim_support(latex, key_results=key_results)
        self.assertEqual(len(report["numeric_unbound_claims"]), 1)
        self.assertIn("99.9", report["numeric_unbound_claims"][0])

    def test_numeric_claim_can_bind_multiple_metrics_in_one_record(self) -> None:
        latex = r"""
        \begin{abstract}
        We improve accuracy to 90.0\% and F1 to 99.9\% (Figure 1).
        \end{abstract}
        """
        key_results = {
            "values": ["0.9", "0.999"],
            "evidence_values": [
                {
                    "value": "0.9",
                    "metric": "accuracy",
                    "record_id": "record-1",
                },
                {
                    "value": "0.999",
                    "metric": "f1",
                    "record_id": "record-1",
                },
            ],
        }
        report = assess_claim_support(latex, key_results=key_results)
        self.assertEqual(report["numeric_unbound_claims"], [])

    def test_numeric_match_accepts_percent_conversion_but_rejects_wrong_rounding(self) -> None:
        self.assertTrue(
            _numeric_token_matches(
                _claim_numeric_tokens(r"accuracy is 12.3\%"), ["0.123"]
            )
        )
        self.assertFalse(
            _numeric_token_matches(
                _claim_numeric_tokens(r"accuracy is 12.3\%"), ["0.124"]
            )
        )
        self.assertFalse(
            _numeric_token_matches(
                _claim_numeric_tokens(r"accuracy is 90.0\%"), ["0.901"]
            )
        )

    def test_numeric_claim_cannot_combine_values_from_different_records(self) -> None:
        claim = r"We improve accuracy to 90.0\% and F1 to 99.9\% (Figure 1)."
        key_results = {
            "evidence_values": [
                {"value": "0.9", "metric": "accuracy", "record_id": "record-1"},
                {"value": "0.999", "metric": "f1", "record_id": "record-2"},
            ]
        }
        matched, detail = _numeric_evidence_matches(
            claim, _claim_numeric_tokens(claim), key_results
        )
        self.assertFalse(matched)
        self.assertEqual(
            detail["reason"], "one_or_more_numbers_not_bound_to_same_result_group"
        )

    def test_rooted_numeric_binding_ignores_forged_cache_and_exploratory_records(
        self,
    ) -> None:
        claim = r"We improve accuracy to 99.9\%."
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "experiment_registry.jsonl").write_text(
                json.dumps(
                    {
                        "record_id": "exploratory-1",
                        "task_id": "task-0",
                        "status": "completed",
                        "study_phase": "exploratory",
                        "metric": "accuracy",
                        "result_summary": {"metric_mean": 0.999},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            matched, detail = _numeric_evidence_matches(
                claim,
                _claim_numeric_tokens(claim),
                {"values": ["0.999"]},
                root,
            )

        self.assertFalse(matched)
        self.assertEqual(detail["reason"], "no_artifact_numeric_values")

    def test_manuscript_claim_uses_dual_marker_to_bind_graph_claim(self) -> None:
        records = [
            {
                "record_id": "record-a",
                "task_id": "task-a",
                "status": "completed",
                "study_phase": "confirmatory",
                "metric": "accuracy",
                "result_summary": {"metric_mean": 0.8},
            }
        ]
        graph = {
            "nodes": [
                {"id": "task-a", "type": "experiment"},
                {"id": "metric-a", "type": "metric", "label": "accuracy"},
                {"id": "claim-a", "type": "claim"},
            ],
            "edges": [
                {"source": "task-a", "target": "metric-a", "type": "supports"},
                {"source": "metric-a", "target": "claim-a", "type": "supports"},
            ],
        }
        latex = (
            r"\begin{abstract}We show accuracy improves. "
            r"\claimref[claim=claim-a]{tree-node-7}\end{abstract}"
        )

        binding = _claim_graph_binding_report(records, graph)
        report = _manuscript_claim_binding_report(latex, records, binding)

        self.assertEqual(report["status"], "ready")
        self.assertEqual(report["bindings"][0]["bound_claim_ids"], ["claim-a"])
        self.assertEqual(report["bindings"][0]["marker_targets"], ["tree-node-7"])

    def test_numeric_claim_and_explicit_graph_claim_must_share_one_path(self) -> None:
        records = [
            {
                "record_id": "record-a",
                "task_id": "task-a",
                "status": "completed",
                "study_phase": "confirmatory",
                "metric": "accuracy",
                "result_summary": {"metric_mean": 0.8},
            },
            {
                "record_id": "record-b",
                "task_id": "task-b",
                "status": "completed",
                "study_phase": "confirmatory",
                "metric": "accuracy",
                "result_summary": {"metric_mean": 0.9},
            },
        ]
        graph = {
            "nodes": [
                {"id": "task-a", "type": "experiment"},
                {"id": "metric-a", "type": "metric", "label": "accuracy"},
                {"id": "claim-a", "type": "claim"},
                {"id": "task-b", "type": "experiment"},
                {"id": "metric-b", "type": "metric", "label": "accuracy"},
                {"id": "claim-b", "type": "claim"},
            ],
            "edges": [
                {"source": "task-a", "target": "metric-a", "type": "supports"},
                {"source": "metric-a", "target": "claim-a", "type": "supports"},
                {"source": "task-b", "target": "metric-b", "type": "supports"},
                {"source": "metric-b", "target": "claim-b", "type": "supports"},
            ],
        }
        latex = (
            r"\begin{abstract}We improve accuracy to 80.0\%"
            r"\claimref[claim=claim-b]{tree-node-9}.\end{abstract}"
        )

        binding = _claim_graph_binding_report(records, graph)
        report = _manuscript_claim_binding_report(latex, records, binding)

        self.assertEqual(report["status"], "blocked")
        self.assertIn(
            "numeric_record_and_graph_claim_ref_disagree",
            report["bindings"][0]["reasons"],
        )

    def test_trailing_claim_marker_is_attached_to_preceding_sentence(self) -> None:
        latex = (
            "\\begin{abstract}\n"
            "We show accuracy improves. \\claimref[claim=claim-a]{tree-node-7}\n"
            "A neutral follow-up sentence.\n"
            "\\end{abstract}"
        )

        ledger = build_claim_evidence_ledger(latex)

        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["claim_refs"], ["tree-node-7"])
        self.assertEqual(
            ledger[0]["claim_markers"][0]["options"]["claim"], "claim-a"
        )

    def test_marked_latex_fragment_is_scanned_without_full_template(self) -> None:
        latex = r"Our method improves accuracy to 80.0\%\claimref{claim-a}."

        ledger = build_claim_evidence_ledger(latex)

        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0]["claim_refs"], ["claim-a"])

    def test_key_results_exclude_exploratory_registry_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = [
                {
                    "record_id": "exploratory-1",
                    "task_id": "task-0",
                    "status": "completed",
                    "study_phase": "exploratory",
                    "result_summary": {"metric_mean": 0.99},
                },
                {
                    "record_id": "confirmatory-1",
                    "task_id": "task-0",
                    "status": "completed",
                    "study_phase": "confirmatory",
                    "result_summary": {"metric_mean": 0.80},
                },
            ]
            (root / "experiment_registry.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            result = extract_key_results(root, "")

        self.assertIn("0.8", result["values"])
        self.assertNotIn("0.99", result["values"])
        self.assertEqual(result["evidence_scope"], "confirmatory_registry")

    def test_claim_prompt_explains_dual_graph_marker(self) -> None:
        guidance = render_claim_prompt_snippet()

        self.assertIn(r"\claimref[claim=<claim_id>]{<node_id>}", guidance)
        self.assertIn("Never invent either ID", guidance)

    def test_claim_graph_requires_every_claim_node_to_be_bound(self) -> None:
        records = [
            {
                "record_id": "record-1",
                "task_id": "task-0",
                "status": "completed",
                "metric": "accuracy",
            }
        ]
        graph = {
            "nodes": [
                {"id": "task-0", "type": "experiment"},
                {"id": "metric-0", "type": "metric", "label": "accuracy"},
                {"id": "claim-0", "type": "claim"},
                {"id": "claim-1", "type": "claim"},
            ],
            "edges": [
                {"source": "task-0", "target": "metric-0", "type": "supports"},
                {"source": "metric-0", "target": "claim-0", "type": "supports"},
            ],
        }
        binding = _claim_graph_binding_report(records, graph)
        self.assertEqual(binding["status"], "blocked")
        self.assertEqual(binding["unbound_claim_ids"], ["claim-1"])

    def test_citation_audit_detects_missing_duplicate_and_unused_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "latex").mkdir()
            latex = r"""
            \cite{known,missing}
            \bibliography{references}
            """
            (root / "latex" / "template.tex").write_text(latex, encoding="utf-8")
            (root / "latex" / "references.bib").write_text(
                "@article{known, title={A}}\n"
                "@article{known, title={B}}\n"
                "@article{unused, title={C}}\n",
                encoding="utf-8",
            )
            report = _citation_integrity_report(latex, root)
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing_keys"], ["missing"])
        self.assertIn("known", report["duplicate_keys"])
        self.assertIn("unused", report["unused_keys"])

    def test_citation_audit_accepts_unique_used_local_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "latex").mkdir()
            latex = r"\cite{known}\bibliography{references}"
            (root / "latex" / "template.tex").write_text(latex, encoding="utf-8")
            (root / "latex" / "references.bib").write_text(
                "@article{known, title={A}}\n", encoding="utf-8"
            )
            report = _citation_integrity_report(latex, root)
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing_keys"], [])
        self.assertEqual(report["unused_keys"], [])

    def test_hash_only_result_manifest_is_not_evidence_for_a_rooted_gate(self) -> None:
        record = {
            "artifacts": {
                "artifact_hashes": {"result": _digest("a")},
            },
            "verification_output_hash": _digest("a"),
        }
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.assertFalse(_result_artifact_manifest_present(record, root))
            self.assertFalse(_verification_output_matches(record, root))

    def test_result_path_escape_is_rejected_by_rooted_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "project"
            outside = Path(td) / "outside.json"
            root.mkdir()
            outside.write_text("result", encoding="utf-8")
            digest = "sha256:" + hashlib.sha256(outside.read_bytes()).hexdigest()
            record = {
                "artifacts": {"result": "../outside.json"},
                "verification_output_hash": digest,
            }
            self.assertFalse(_result_artifact_manifest_present(record, root))
            self.assertFalse(_verification_output_matches(record, root))

    def test_verification_output_hash_must_bind_to_evaluator_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            input_path = root / "input.json"
            result_path = root / "result.json"
            input_path.write_text("input", encoding="utf-8")
            result_path.write_text("result", encoding="utf-8")
            input_hash = "sha256:" + hashlib.sha256(input_path.read_bytes()).hexdigest()
            result_hash = (
                "sha256:" + hashlib.sha256(result_path.read_bytes()).hexdigest()
            )
            record = {
                "artifacts": {"input": str(input_path), "result": str(result_path)},
                "evaluator_result_hash": result_hash,
                "verification_output_hash": input_hash,
            }

            self.assertFalse(_verification_output_matches(record, root))
            record["verification_output_hash"] = result_hash
            self.assertTrue(_verification_output_matches(record, root))

    def test_malformed_registry_row_blocks_publication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "experiment_registry.jsonl").write_text(
                '{"record_id":"ok"}\nnot-json\n', encoding="utf-8"
            )
            gate = build_scientific_evidence_gate(root)
        self.assertIn("registry_parse_integrity", gate["hard_failures"])

    def test_registry_numeric_results_include_zero_without_scraping_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "experiment_registry.jsonl").write_text(
                json.dumps(
                    {
                        "record_id": "task_0_run_0",
                        "task_id": "task_0",
                        "status": "completed",
                        "study_phase": "confirmatory",
                        "metric": "accuracy",
                        "result_summary": {
                            "metric_mean": 0,
                            "baseline_metric_mean": 0.2,
                            "delta_vs_baseline": -0.2,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            result = extract_key_results(root, "")

        self.assertIn("0", result["values"])
        self.assertIn("0.2", result["values"])
        self.assertNotIn("0_run_0", result["values"])

    def test_high_prose_scores_cannot_promote_an_unverified_evidence_chain(
        self,
    ) -> None:
        report = {
            "professional": {"overall": {"score": 5.0}},
            "rigor": {"score": 5.0},
            "claim_support": {"score": 5.0},
            "claim_alignment": {"score": 5.0},
            "numeric_coverage": {"score": 5.0},
            "scientific_evidence": {
                "status": "blocked",
                "hard_failures": ["independent_verification"],
            },
        }
        self.assertFalse(
            _quality_gate_passed(
                report,
                quality_threshold=4.0,
                rigor_threshold=3.5,
                claim_support_threshold=3.5,
            )
        )

    def test_prose_cannot_pass_rigor_without_registry(self) -> None:
        latex = r"""
        \begin{abstract}Our method significantly improves accuracy by 12.3%.\end{abstract}
        baseline, ablation, p-value, confidence interval, seed, reproducibility.
        \begin{figure}\caption{Results}\label{fig:results}\end{figure}
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rigor = assess_experiment_rigor(root, latex)
            claims = assess_claim_support(latex, root)

        self.assertEqual(rigor["score"], 1.0)
        self.assertIn("experiment_registry_missing", rigor["hard_failures"])
        self.assertLessEqual(claims["score"], 2.0)
        self.assertEqual(claims["artifact_binding"]["status"], "blocked")

    def test_claim_numbers_accept_percent_rendering_of_artifact_decimal(self) -> None:
        latex = r"""
        \begin{abstract}We improve accuracy by 12.3\% (Figure 2).</abstract>
        """
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Deliberately provide the graph/registry so only numeric binding is
            # exercised; the publication gate itself remains blocked here.
            (root / "experiment_registry.jsonl").write_text(
                json.dumps(
                    {
                        "record_id": "record-1",
                        "task_id": "task_0",
                        "status": "completed",
                        "study_phase": "confirmatory",
                        "metric": "accuracy",
                        "result_summary": {"metric_mean": 0.123},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "claim_evidence_graph.json").write_text(
                json.dumps(
                    {
                        "nodes": [
                            {"id": "task_0", "type": "experiment"},
                            {"id": "metric_0", "type": "metric"},
                            {"id": "claim_0", "type": "claim"},
                        ],
                        "edges": [
                            {
                                "source": "task_0",
                                "target": "metric_0",
                                "type": "supports",
                            },
                            {
                                "source": "metric_0",
                                "target": "claim_0",
                                "type": "supports",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report = assess_claim_support(
                latex,
                root,
                {"values": ["0.123"]},
            )

        self.assertEqual(report["numeric_unbound_claims"], [])

    def test_gate_requires_locked_contract_and_independent_verification(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            gate = build_scientific_evidence_gate(root)
        self.assertEqual(gate["status"], "blocked")
        self.assertIn("preregistration_present", gate["hard_failures"])
        self.assertIn("independent_verification", gate["hard_failures"])

    def test_malformed_preregistration_is_reported_as_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "preregistration.json").write_text(
                json.dumps(
                    {
                        "outcomes": "not-a-list",
                        "analysis_plan": "not-an-object",
                        "data_policy": "not-an-object",
                    }
                ),
                encoding="utf-8",
            )
            gate = build_scientific_evidence_gate(root)

        self.assertEqual(gate["status"], "blocked")
        self.assertIn("locked_preregistration", gate["hard_failures"])

    def test_gate_passes_only_for_artifact_bound_verified_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            locked = lock_preregistration(
                build_preregistration(_idea(), _plan()),
                split_hashes={"task_0": _digest("a")},
                registered_by="planner",
            )
            (root / "preregistration.json").write_text(
                json.dumps(locked), encoding="utf-8"
            )
            records = []
            for seed in (11, 22, 33):
                result_summary = {
                    "metric_mean": 0.82,
                    "effect_size": 0.12,
                    "baseline_metric_mean": 0.70,
                    "delta_vs_baseline": 0.12,
                }
                input_hash = _write_hashed_json(
                    root / f"input-{seed}.json", {"seed": seed}
                )
                result_hash = _write_hashed_json(
                    root / f"result-{seed}.json", result_summary
                )
                records.append(
                    {
                        "record_id": f"run-{seed}",
                        "task_id": "task_0",
                        "dataset": "benchmark-v1",
                        "metric": "accuracy",
                        "baseline_ref": "baseline-a",
                        "status": "completed",
                        "study_phase": "confirmatory",
                        "seed": seed,
                        "preregistration_id": locked["preregistration_id"],
                        "protocol_fidelity_hash": _protocol_fidelity_hash(
                            locked, "task_0"
                        ),
                        "record_id": f"run-{seed}",
                        "producer_id": "experiment-agent",
                        "finished_at": "2026-01-01T00:00:00+00:00",
                        "result_summary": result_summary,
                        "dataset_split_hash": _digest("a"),
                        "metric_provenance": "deterministic_verified",
                        "evaluator_input_hash": input_hash,
                        "evaluator_result_hash": result_hash,
                        "holdout_access": "verifier_only",
                        "artifacts": {
                            "input": str(root / f"input-{seed}.json"),
                            "result": str(root / f"result-{seed}.json"),
                            "artifact_hashes": {
                                "input": input_hash,
                                "result": result_hash,
                            },
                        },
                        "verification_recomputed": True,
                        "verification_metric_hash": _canonical_hash(result_summary),
                        "verification_output_hash": result_hash,
                        "verification_command": "python verify_results.py",
                    }
                )
            (root / "experiment_registry.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            (root / "claim_evidence_graph.json").write_text(
                json.dumps(
                    {
                        "nodes": [
                            {"id": "task_0", "type": "experiment"},
                            {"id": "task_0_metric", "type": "metric"},
                            {"id": "claim_0", "type": "claim"},
                        ],
                        "edges": [
                            {
                                "source": "task_0",
                                "target": "task_0_metric",
                                "type": "supports",
                            },
                            {
                                "source": "task_0_metric",
                                "target": "claim_0",
                                "type": "supports",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            reproduction_summary = {
                "metric_mean": 0.81,
                "baseline_metric_mean": 0.70,
                "delta_vs_baseline": 0.11,
                "effect_size": 0.11,
            }
            reproduction_input_hash = _write_hashed_json(
                root / "reproduction-input.json", {"seed": 44}
            )
            reproduction_result_hash = _write_hashed_json(
                root / "reproduction-result.json", reproduction_summary
            )
            records.append(
                {
                    "record_id": "reproduction-1",
                    "independent_reproduction": True,
                    "replicates_record_id": "run-11",
                    "task_id": "task_0",
                    "dataset": "benchmark-v1",
                    "metric": "accuracy",
                    "baseline_ref": "baseline-a",
                    "preregistration_id": locked["preregistration_id"],
                    "protocol_fidelity_hash": _protocol_fidelity_hash(locked, "task_0"),
                    "dataset_split_hash": _digest("a"),
                    "holdout_access": "verifier_only",
                    "producer_id": "reproduction-agent",
                    "clean_room": True,
                    "verifier_id": "verification-agent",
                    "status": "completed",
                    "finished_at": "2026-01-02T00:00:00+00:00",
                    "artifacts": {
                        "input": str(root / "reproduction-input.json"),
                        "result": str(root / "reproduction-result.json"),
                        "artifact_hashes": {
                            "input": reproduction_input_hash,
                            "result": reproduction_result_hash,
                        },
                    },
                    "result_summary": reproduction_summary,
                    "verification_recomputed": True,
                    "verification_metric_hash": _canonical_hash(reproduction_summary),
                    "verification_output_hash": reproduction_result_hash,
                    "evaluator_input_hash": reproduction_input_hash,
                    "evaluator_result_hash": reproduction_result_hash,
                    "verification_command": "python verify_results.py",
                }
            )
            (root / "experiment_registry.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            (root / "latex").mkdir()
            (root / "latex" / "template.tex").write_text(
                r"""
                \begin{abstract}
                Our method improves accuracy to 82.0\% \claimref{claim_0}.\cite{ref}
                \end{abstract}
                \section{Results}
                \bibliography{references}
                """,
                encoding="utf-8",
            )
            (root / "latex" / "references.bib").write_text(
                "@article{ref, title={A reference}}\n", encoding="utf-8"
            )
            save_experiment_registry(root, records)
            save_evidence_snapshot(
                root, build_evidence_snapshot(root, records=records)
            )
            verification = build_verification_report(
                locked,
                records,
                verifier_id="verification-agent",
                clean_room=True,
                verification_root=root,
            )
            (root / "verification_report.json").write_text(
                json.dumps(verification),
                encoding="utf-8",
            )

            gate = build_scientific_evidence_gate(root)

        self.assertEqual(gate["status"], "verified")
        self.assertEqual(gate["hard_failures"], [])
        self.assertTrue(gate["submission_ready"])


if __name__ == "__main__":
    unittest.main()
