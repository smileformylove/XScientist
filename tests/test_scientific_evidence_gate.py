from __future__ import annotations

import json
import hashlib
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.utils.high_quality_pipeline import (
    _build_submission_readiness,
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
    canonical_hash,
    save_evidence_snapshot,
)
from ai_scientist.utils.data_readiness import prepare_data_contract
from ai_scientist.utils.experiment_registry import save_experiment_registry
from ai_scientist.utils.claim_registry import render_claim_prompt_snippet
from ai_scientist.utils.research_integrity import (
    _canonical_hash,
    _protocol_fidelity_hash,
    _result_artifact_manifest_present,
    _verification_output_matches,
    build_verification_report,
    build_preregistration,
    derive_adaptive_state_hashes,
    lock_preregistration,
)
from xscientist.research_commands import (
    save_hypothesis as save_research_hypothesis,
    save_preregistration as save_research_preregistration,
)
from xscientist.research_vcs import ResearchRepository


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
    def _host_attested_preregistration(
        self, root: Path
    ) -> tuple[Path, dict[str, object]]:
        if shutil.which("git") is None:
            self.skipTest("Git is not installed")
        repository = ResearchRepository.init(root, question="Fixed question")
        hypothesis = save_research_hypothesis(
            str(root),
            statement="Method A improves the fixed baseline",
            falsifier="delta <= 0",
        )
        registration = save_research_preregistration(
            str(root),
            hypothesis_id=hypothesis["object"].object_id,
            dataset="benchmark-v1",
            metric="accuracy",
            baseline="baseline-a",
            split_hash=_digest("a"),
            registered_by="lead-researcher",
            minimum_effect=0.01,
            minimum_seeds=3,
        )
        locked = repository.get(registration["object"].object_id)["payload"]
        paper_root = root / "03_papers" / "candidate"
        paper_root.mkdir(parents=True)
        (paper_root / "preregistration.json").write_text(
            json.dumps(locked), encoding="utf-8"
        )
        return paper_root, locked

    def _prepare_empirical_data(self, root: Path) -> dict[str, object]:
        source = root.parent / f"{root.name}-private-data"
        source.mkdir()
        (source / "observations.csv").write_text("x,y\n1,2\n", encoding="utf-8")
        return prepare_data_contract(root, data_dir=source, required=True)

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

    def test_numeric_match_accepts_percent_conversion_but_rejects_wrong_rounding(
        self,
    ) -> None:
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
        self.assertEqual(ledger[0]["claim_markers"][0]["options"]["claim"], "claim-a")

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

    def test_gate_rejects_symlinked_critical_artifact_leaves(self) -> None:
        cases = [
            (
                Path("latex") / "template.tex",
                "latex_artifact_boundary",
                None,
                b"\\documentclass{article}\n",
            ),
            (
                Path("preregistration.json"),
                "preregistration_artifact_boundary",
                None,
                b"{}",
            ),
            (
                Path("experiment_registry.jsonl"),
                "experiment_registry_artifact_boundary",
                None,
                b"{}\n",
            ),
            (
                Path("claim_evidence_graph.json"),
                "claim_graph_artifact_boundary",
                None,
                b"{}",
            ),
            (
                Path("verification_report.json"),
                "verification_report_artifact_boundary",
                None,
                b"{}",
            ),
            (
                Path("evidence_snapshot.json"),
                "evidence_snapshot_artifact_boundary",
                None,
                b"{}",
            ),
            (
                Path("verifier_authority_receipt.json"),
                "verifier_authority_artifact_boundary",
                "neurips",
                b"{}",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            for index, (relative, check_id, venue, content) in enumerate(cases):
                with self.subTest(relative=relative):
                    paper_root = base / f"paper-{index}"
                    paper_root.mkdir()
                    outside = base / f"outside-{index}"
                    outside.write_bytes(content)
                    linked = paper_root / relative
                    linked.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        linked.symlink_to(outside)
                    except OSError as exc:  # pragma: no cover - platform capability
                        self.skipTest(f"file symlinks unavailable: {exc}")

                    gate = build_scientific_evidence_gate(
                        paper_root,
                        target_venue=venue,
                    )
                    checks = {item["id"]: item for item in gate["checks"]}
                    self.assertFalse(checks[check_id]["passed"])
                    self.assertIn(check_id, gate["hard_failures"])
                    self.assertIn("symlink_rejected", checks[check_id]["detail"])

    def test_gate_rejects_oversized_artifact_and_registry_row_flood(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            oversized_root = Path(td) / "oversized"
            oversized_root.mkdir()
            with (oversized_root / "preregistration.json").open("wb") as handle:
                handle.truncate(16 * 1024 * 1024 + 1)
            oversized_gate = build_scientific_evidence_gate(oversized_root)

            row_root = Path(td) / "rows"
            row_root.mkdir()
            (row_root / "experiment_registry.jsonl").write_text(
                "{}\n" * 100_001,
                encoding="utf-8",
            )
            row_gate = build_scientific_evidence_gate(row_root)

        self.assertIn(
            "preregistration_artifact_boundary",
            oversized_gate["hard_failures"],
        )
        self.assertIn("registry_parse_integrity", row_gate["hard_failures"])
        row_check = {item["id"]: item for item in row_gate["checks"]}[
            "registry_parse_integrity"
        ]
        self.assertIn("row_limit_exceeded", row_check["detail"])

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

    def test_high_scores_cannot_hide_missing_ablation_or_robustness_evidence(
        self,
    ) -> None:
        report = {
            "professional": {"overall": {"score": 5.0}},
            "rigor": {
                "score": 5.0,
                "hard_failures": [
                    "registry_ablation_missing",
                    "registry_robustness_missing",
                ],
            },
            "claim_support": {"score": 5.0},
            "claim_alignment": {"score": 5.0},
            "numeric_coverage": {"score": 5.0},
            "scientific_evidence": {"status": "verified", "hard_failures": []},
        }
        self.assertFalse(
            _quality_gate_passed(
                report,
                quality_threshold=4.0,
                rigor_threshold=3.5,
                claim_support_threshold=3.5,
            )
        )
        readiness = _build_submission_readiness(
            report,
            paper_type="normal",
            target_venue="icml",
            quality_threshold=4.0,
            rigor_threshold=3.5,
            claim_support_threshold=3.5,
        )
        self.assertFalse(readiness["ready"])
        self.assertEqual(readiness["decision"], "blocked")
        self.assertIn(
            "rigor gate: registry_ablation_missing",
            readiness["blockers"],
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

    def test_publication_gate_rejects_unanchored_legacy_registry_history(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save_experiment_registry(
                root,
                [{"record_id": "legacy-run", "status": "completed"}],
            )
            integrity = json.loads(
                (root / "experiment_registry.integrity.json").read_text(
                    encoding="utf-8"
                )
            )
            core = {
                "record_count": integrity["row_count"],
                "records_hash": integrity["records_hash"],
                "raw_hash": integrity["raw_hash"],
                "chain_tip": integrity["chain_tip"],
            }
            legacy_row = {
                "version": 1,
                **core,
                "audit_hash": canonical_hash(core),
            }
            (root / "experiment_registry.history.jsonl").write_text(
                json.dumps(legacy_row) + "\n",
                encoding="utf-8",
            )

            gate = build_scientific_evidence_gate(root)

        checks = {item["id"]: item for item in gate["checks"]}
        self.assertFalse(checks["registry_append_only_integrity"]["passed"])
        self.assertIn(
            "legacy_history_unanchored",
            checks["registry_append_only_integrity"]["detail"],
        )
        self.assertIn("registry_append_only_integrity", gate["hard_failures"])

    def test_unknown_target_venue_cannot_relax_publication_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            gate = build_scientific_evidence_gate(
                Path(td),
                target_venue="neuripss",
            )

        self.assertIn("target_venue_valid", gate["hard_failures"])
        checks = {item["id"]: item for item in gate["checks"]}
        self.assertFalse(checks["target_venue_valid"]["passed"])

    def test_top_conference_gate_requires_exploration_confirmation_isolation(
        self,
    ) -> None:
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

            general_gate = build_scientific_evidence_gate(root)
            top_gate = build_scientific_evidence_gate(root, target_venue="icml")

        self.assertNotIn(
            "adaptive_exploration_state_freeze", general_gate["hard_failures"]
        )
        self.assertNotIn(
            "confirmatory_frozen_state_fidelity", general_gate["hard_failures"]
        )
        self.assertIn("adaptive_exploration_state_freeze", top_gate["hard_failures"])
        self.assertIn("confirmatory_frozen_state_fidelity", top_gate["hard_failures"])
        self.assertIn("venue_template_attestation", top_gate["hard_failures"])
        self.assertIn("independent_verifier_authority", top_gate["hard_failures"])
        self.assertNotIn(
            "independent_verifier_authority", general_gate["hard_failures"]
        )

    def test_top_conference_gate_invokes_external_verifier_authority(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "verifier_authority_receipt.json").write_text(
                "{}", encoding="utf-8"
            )
            with patch(
                "ai_scientist.utils.high_quality_pipeline."
                "verify_verifier_authority_receipt",
                return_value={
                    "ok": True,
                    "status": "verified",
                    "errors": [],
                    "verifier_identity": "human:external-reviewer",
                },
            ) as verifier:
                gate = build_scientific_evidence_gate(
                    root,
                    target_venue="neurips",
                    verifier_trust_store="/outside/workspace/trust.json",
                )

        checks = {item["id"]: item for item in gate["checks"]}
        self.assertTrue(checks["independent_verifier_authority"]["passed"])
        verifier.assert_called_once()
        self.assertEqual(verifier.call_args.args[3], "/outside/workspace/trust.json")

    def test_top_conference_gate_accepts_valid_host_state_freeze_contract(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "study"
            paper_root, _locked = self._host_attested_preregistration(project_root)
            self._prepare_empirical_data(project_root)
            gate = build_scientific_evidence_gate(paper_root, target_venue="neurips")

        checks = {item["id"]: item for item in gate["checks"]}
        self.assertTrue(checks["adaptive_exploration_state_freeze"]["passed"])
        self.assertTrue(checks["research_vcs_attestation"]["passed"])
        self.assertTrue(checks["empirical_data_attestation"]["passed"])
        self.assertNotIn("adaptive_exploration_state_freeze", gate["hard_failures"])
        self.assertNotIn("research_vcs_attestation", gate["hard_failures"])
        self.assertNotIn("empirical_data_attestation", gate["hard_failures"])
        self.assertIn("confirmatory_frozen_state_fidelity", gate["hard_failures"])

    def test_top_conference_gate_rejects_nonexistent_research_vcs_head(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "study"
            ResearchRepository.init(project_root, question="Fixed question")
            paper_root = project_root / "03_papers" / "candidate"
            paper_root.mkdir(parents=True)
            draft = build_preregistration(_idea(), _plan())
            draft["data_policy"]["split_hashes"] = {"task_0": _digest("a")}
            fake_head = "f" * 40
            locked = lock_preregistration(
                draft,
                split_hashes={"task_0": _digest("a")},
                registered_by="planner",
                freeze_inputs={
                    "research_vcs_head": fake_head,
                    **derive_adaptive_state_hashes(draft, research_vcs_head=fake_head),
                },
            )
            (paper_root / "preregistration.json").write_text(
                json.dumps(locked), encoding="utf-8"
            )
            self._prepare_empirical_data(project_root)
            gate = build_scientific_evidence_gate(paper_root, target_venue="icml")

        checks = {item["id"]: item for item in gate["checks"]}
        self.assertTrue(checks["adaptive_exploration_state_freeze"]["passed"])
        self.assertFalse(checks["research_vcs_attestation"]["passed"])
        self.assertIn("research_vcs_attestation", gate["hard_failures"])
        self.assertIn(
            "research_vcs_head_not_resolvable",
            gate["research_vcs_attestation"]["errors"],
        )

    def test_top_conference_gate_requires_committed_preregistration_transition(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "study"
            repository = ResearchRepository.init(
                project_root, question="Fixed question"
            )
            paper_root = project_root / "03_papers" / "candidate"
            paper_root.mkdir(parents=True)
            draft = build_preregistration(_idea(), _plan())
            draft["data_policy"]["split_hashes"] = {"task_0": _digest("a")}
            real_head = str(repository.status()["head"])
            locked = lock_preregistration(
                draft,
                split_hashes={"task_0": _digest("a")},
                registered_by="planner",
                freeze_inputs={
                    "research_vcs_head": real_head,
                    **derive_adaptive_state_hashes(
                        draft,
                        research_vcs_head=real_head,
                    ),
                },
            )
            (paper_root / "preregistration.json").write_text(
                json.dumps(locked), encoding="utf-8"
            )
            self._prepare_empirical_data(project_root)
            gate = build_scientific_evidence_gate(
                paper_root,
                target_venue="neurips",
            )

        self.assertIn("research_vcs_attestation", gate["hard_failures"])
        self.assertIn(
            "research_vcs_preregistration_object_missing_or_ambiguous",
            gate["research_vcs_attestation"]["errors"],
        )

    def test_top_conference_gate_recomputes_component_hashes_from_host_state(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "study"
            repository = ResearchRepository.init(
                project_root, question="Fixed question"
            )
            paper_root = project_root / "03_papers" / "candidate"
            paper_root.mkdir(parents=True)
            draft = build_preregistration(_idea(), _plan())
            draft["data_policy"]["split_hashes"] = {"task_0": _digest("a")}
            real_head = str(repository.status()["head"])
            derived = derive_adaptive_state_hashes(
                draft,
                research_vcs_head=real_head,
            )
            invented = {
                "code_state_hash": _digest("b"),
                "memory_state_hash": _digest("c"),
                "evaluator_spec_hash": _digest("d"),
            }
            invented["research_state_hash"] = _canonical_hash(
                {"kind": "confirmatory_research_state", **invented}
            )
            locked = lock_preregistration(
                draft,
                split_hashes={"task_0": _digest("a")},
                registered_by="planner",
                freeze_inputs={
                    "research_vcs_head": real_head,
                    **derived,
                },
            )
            locked["adaptive_state_freeze"].update(invented)
            locked["adaptive_state_freeze"]["research_state_hash"] = invented[
                "research_state_hash"
            ]
            (paper_root / "preregistration.json").write_text(
                json.dumps(locked), encoding="utf-8"
            )
            self._prepare_empirical_data(project_root)
            gate = build_scientific_evidence_gate(paper_root, target_venue="neurips")

        self.assertIn("research_vcs_attestation", gate["hard_failures"])
        self.assertIn(
            "research_vcs_code_state_hash_mismatch",
            gate["research_vcs_attestation"]["errors"],
        )

    def test_top_conference_gate_rejects_explicit_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "study"
            paper_root, _locked = self._host_attested_preregistration(project_root)
            prepare_data_contract(project_root, allow_synthetic=True, required=True)
            gate = build_scientific_evidence_gate(paper_root, target_venue="neurips")

        self.assertIn("empirical_data_attestation", gate["hard_failures"])
        self.assertIn(
            "synthetic_data_not_empirical",
            gate["empirical_data_attestation"]["errors"],
        )

    def test_top_conference_gate_rejects_tampered_data_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            project_root = Path(td) / "study"
            paper_root, _locked = self._host_attested_preregistration(project_root)
            manifest = self._prepare_empirical_data(project_root)
            snapshot_file = (
                project_root
                / ".ara-store"
                / "datasets"
                / str(manifest["snapshot_id"]).removeprefix("sha256:")
                / "observations.csv"
            )
            snapshot_file.chmod(0o644)
            snapshot_file.write_text("x,y\n9,9\n", encoding="utf-8")
            gate = build_scientific_evidence_gate(paper_root, target_venue="icml")

        self.assertIn("empirical_data_attestation", gate["hard_failures"])
        self.assertIn(
            "data_snapshot_file_hash_mismatch",
            gate["empirical_data_attestation"]["errors"],
        )

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
                    "standard_error": 0.01,
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
            save_experiment_registry(root, records)
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
                "confidence_interval": [0.08, 0.14],
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
            save_experiment_registry(root, records)
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
            save_evidence_snapshot(root, build_evidence_snapshot(root, records=records))
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
