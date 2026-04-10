from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ai_scientist.utils.readiness_benchmark import (
    build_readiness_benchmark,
    export_readiness_benchmark_markdown,
)
from ai_scientist.utils.pipeline_contracts import (
    initialize_pipeline_contracts,
    record_fallback_event,
    save_contract_artifact,
)
from research_manager import ResearchManager


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_quality_run(
    research_root: Path,
    folder_name: str,
    *,
    idea_name: str,
    target_venue: str,
    ready: bool,
    gate_passed: bool,
    submission_priority: float,
    quality_score: float,
    rigor_score: float,
    claim_support_score: float,
    claim_alignment_score: float,
    numeric_coverage_score: float,
    breakthrough_score: float,
    evidence_density_score: float,
    contribution_count: int,
    blockers: list[str] | None = None,
    categories: dict[str, int] | None = None,
    unsupported_claims_count: int = 0,
    fallback_count: int = 0,
    strict_fallback_count: int = 0,
    stage_standards: dict | None = None,
    self_evolution: dict | None = None,
    process_alignment: dict | None = None,
) -> Path:
    run_dir = research_root / "papers" / folder_name
    initialize_pipeline_contracts(run_dir)
    _write_json(
        run_dir / "idea.json",
        {
            "Name": idea_name,
            "Title": f"{idea_name} title",
        },
    )
    _write_json(
        run_dir / "quality" / "high_quality_result.json",
        {
            "status": "success",
            "target_venue": target_venue,
            "quality_score_after": quality_score,
            "rigor_score_after": rigor_score,
            "claim_support_after": claim_support_score,
            "claim_alignment_after": claim_alignment_score,
            "numeric_coverage_after": numeric_coverage_score,
            "breakthrough_score": breakthrough_score,
            "evidence_density_score": evidence_density_score,
            "contribution_count": contribution_count,
            "quality_gate_passed": gate_passed,
            "submission_priority_score": submission_priority,
            "submission_priority_tier": "submit_now" if submission_priority >= 88 else "near_ready",
            "blocker_count": len(blockers or []),
            "unsupported_claims_count": unsupported_claims_count,
            "critical_revision_actions_count": 0,
            "rewrite_trace": [{"round": 1}] if gate_passed else [],
            "rewrite_applied": gate_passed,
            "submission_readiness": {
                "status": "ready" if ready else "needs_work",
                "ready": ready,
                "blockers": blockers or [],
                "categories": categories or {},
            },
        },
    )
    for idx in range(fallback_count):
        record_fallback_event(
            run_dir,
            stage="idea_ranking",
            producer="test_readiness_benchmark",
            fallback_kind="heuristic_ranking",
            reason=f"fallback event {idx}",
        )
    for idx in range(strict_fallback_count):
        record_fallback_event(
            run_dir,
            stage="quality_review",
            producer="test_readiness_benchmark",
            fallback_kind="auto_improvement_rewrite",
            reason=f"strict fallback event {idx}",
            strict=True,
        )
    if stage_standards is not None:
        save_contract_artifact(
            run_dir,
            "stage_standards",
            stage_standards,
            producer="test_readiness_benchmark",
        )
    if self_evolution is not None:
        save_contract_artifact(
            run_dir,
            "self_evolution",
            self_evolution,
            producer="test_readiness_benchmark",
        )
    if process_alignment is not None:
        save_contract_artifact(
            run_dir,
            "process_alignment",
            process_alignment,
            producer="test_readiness_benchmark",
        )
    return run_dir


class ReadinessBenchmarkTests(unittest.TestCase):
    def test_build_readiness_benchmark_should_rank_ready_nature_runs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_alpha",
                idea_name="Alpha Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=95.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.2,
                claim_alignment_score=4.1,
                numeric_coverage_score=4.0,
                breakthrough_score=4.4,
                evidence_density_score=3.8,
                contribution_count=3,
            )
            _create_quality_run(
                research_root,
                "paper_beta",
                idea_name="Beta Nature",
                target_venue="nature",
                ready=False,
                gate_passed=False,
                submission_priority=61.0,
                quality_score=4.1,
                rigor_score=3.7,
                claim_support_score=3.8,
                claim_alignment_score=3.5,
                numeric_coverage_score=3.6,
                breakthrough_score=3.1,
                evidence_density_score=2.2,
                contribution_count=2,
                blockers=[
                    "breakthrough potential below Nature-style bar (3.10 < 4.00)",
                    "evidence density below target (2.20 < 3.00)",
                ],
                categories={"breakthrough": 1, "evidence": 1},
                unsupported_claims_count=1,
            )
            _create_quality_run(
                research_root,
                "paper_gamma",
                idea_name="Gamma NeurIPS",
                target_venue="neurips",
                ready=True,
                gate_passed=True,
                submission_priority=98.0,
                quality_score=4.6,
                rigor_score=4.3,
                claim_support_score=4.2,
                claim_alignment_score=4.1,
                numeric_coverage_score=4.2,
                breakthrough_score=4.5,
                evidence_density_score=4.0,
                contribution_count=3,
            )

            benchmark = build_readiness_benchmark(
                research_root,
                target_venue="nature",
                top_n=5,
                max_entries=10,
            )

        self.assertEqual(benchmark["summary"]["entries"], 2)
        self.assertEqual(benchmark["summary"]["venue_match_count"], 2)
        self.assertEqual(benchmark["summary"]["ready_count"], 1)
        self.assertEqual(benchmark["summary"]["gate_pass_count"], 1)
        self.assertEqual(benchmark["summary"]["avg_fallback_count"], 0.0)
        self.assertEqual(benchmark["summary"]["avg_strict_fallback_count"], 0.0)
        self.assertEqual(
            [row["name"] for row in benchmark["ranked_papers"]],
            ["Alpha Nature", "Beta Nature"],
        )
        self.assertGreater(
            benchmark["ranked_papers"][0]["benchmark_score"],
            benchmark["ranked_papers"][1]["benchmark_score"],
        )
        self.assertEqual(
            benchmark["ranked_papers"][1]["failing_metrics"][0]["name"],
            "breakthrough",
        )
        self.assertEqual(
            benchmark["summary"]["top_gap_dimensions"]["breakthrough"],
            1,
        )

    def test_build_readiness_benchmark_should_include_other_venues_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_alpha",
                idea_name="Alpha Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=95.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.2,
                claim_alignment_score=4.1,
                numeric_coverage_score=4.0,
                breakthrough_score=4.4,
                evidence_density_score=3.8,
                contribution_count=3,
            )
            _create_quality_run(
                research_root,
                "paper_gamma",
                idea_name="Gamma NeurIPS",
                target_venue="neurips",
                ready=True,
                gate_passed=True,
                submission_priority=99.0,
                quality_score=4.6,
                rigor_score=4.3,
                claim_support_score=4.2,
                claim_alignment_score=4.1,
                numeric_coverage_score=4.2,
                breakthrough_score=4.5,
                evidence_density_score=4.0,
                contribution_count=3,
            )

            benchmark = build_readiness_benchmark(
                research_root,
                target_venue="nature",
                include_other_venues=True,
                top_n=5,
            )

        self.assertEqual(benchmark["summary"]["entries"], 2)
        self.assertEqual(benchmark["summary"]["venue_match_count"], 1)
        self.assertEqual(benchmark["ranked_papers"][0]["name"], "Alpha Nature")
        self.assertFalse(benchmark["ranked_papers"][1]["venue_match"])

    def test_build_readiness_benchmark_should_penalize_fallback_debt(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_clean",
                idea_name="Clean Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
            )
            _create_quality_run(
                research_root,
                "paper_fallback",
                idea_name="Fallback Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                fallback_count=3,
                strict_fallback_count=1,
            )

            benchmark = build_readiness_benchmark(
                research_root,
                target_venue="nature",
                top_n=5,
                max_entries=10,
            )

        self.assertEqual(benchmark["summary"]["entries"], 2)
        self.assertEqual(benchmark["summary"]["avg_fallback_count"], 2.0)
        self.assertEqual(benchmark["summary"]["avg_strict_fallback_count"], 0.5)
        self.assertEqual(
            benchmark["summary"]["top_fallback_kinds"]["heuristic_ranking"],
            3,
        )
        self.assertEqual(
            [row["name"] for row in benchmark["ranked_papers"][:2]],
            ["Clean Nature", "Fallback Nature"],
        )
        self.assertGreater(
            benchmark["ranked_papers"][0]["benchmark_score"],
            benchmark["ranked_papers"][1]["benchmark_score"],
        )
        self.assertEqual(benchmark["ranked_papers"][1]["fallback_count"], 4)
        self.assertEqual(benchmark["ranked_papers"][1]["strict_fallback_count"], 1)

    def test_build_readiness_benchmark_should_penalize_stage_standard_blockers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_clean_stage",
                idea_name="Clean Stage Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                stage_standards={
                    "schema_version": 1,
                    "overall_score": 93.0,
                    "ready_stage_count": 6,
                    "blocked_stage_count": 0,
                    "needs_attention_stage_count": 0,
                    "missing_stage_count": 0,
                    "summary": {
                        "blocked_stages": [],
                        "attention_stages": [],
                        "missing_stages": [],
                        "top_risks": [],
                    },
                    "stage_results": [],
                },
            )
            _create_quality_run(
                research_root,
                "paper_blocked_stage",
                idea_name="Blocked Stage Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                stage_standards={
                    "schema_version": 1,
                    "overall_score": 61.0,
                    "ready_stage_count": 4,
                    "blocked_stage_count": 1,
                    "needs_attention_stage_count": 1,
                    "missing_stage_count": 0,
                    "summary": {
                        "blocked_stages": ["review"],
                        "attention_stages": ["manuscript"],
                        "missing_stages": [],
                        "top_risks": ["review_evidence_gap"],
                    },
                    "stage_results": [],
                },
            )

            benchmark = build_readiness_benchmark(
                research_root,
                target_venue="nature",
                top_n=5,
                max_entries=10,
            )

        self.assertEqual(
            [row["name"] for row in benchmark["ranked_papers"][:2]],
            ["Clean Stage Nature", "Blocked Stage Nature"],
        )
        self.assertGreater(
            benchmark["ranked_papers"][0]["benchmark_score"],
            benchmark["ranked_papers"][1]["benchmark_score"],
        )
        self.assertEqual(benchmark["ranked_papers"][1]["blocked_stage_count"], 1)
        self.assertIn("review_evidence_gap", benchmark["summary"]["top_stage_standard_risks"])
        self.assertEqual(benchmark["summary"]["avg_blocked_stage_count"], 0.5)
        self.assertEqual(benchmark["summary"]["avg_attention_stage_count"], 0.5)

    def test_build_readiness_benchmark_should_penalize_blocked_self_evolution(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_clean_evolution",
                idea_name="Clean Evolution Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                self_evolution={
                    "summary": {
                        "status": "ready",
                        "score": 92.0,
                        "dominant_lane": "section_rewrite",
                        "dominant_role": "clarity",
                    },
                    "self_check": {
                        "status": "ready",
                        "score": 92.0,
                        "required_failures": [],
                    },
                    "stage_risks": ["clarity_gap"],
                },
            )
            _create_quality_run(
                research_root,
                "paper_blocked_evolution",
                idea_name="Blocked Evolution Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                self_evolution={
                    "summary": {
                        "status": "blocked",
                        "score": 54.0,
                        "dominant_lane": "triage",
                        "dominant_role": "rigor",
                    },
                    "self_check": {
                        "status": "blocked",
                        "score": 54.0,
                        "required_failures": ["repair_targeting", "verification_path"],
                    },
                    "stage_risks": ["repair_ownership_gap", "rigor_validation_gap"],
                },
            )

            benchmark = build_readiness_benchmark(
                research_root,
                target_venue="nature",
                top_n=5,
                max_entries=10,
            )

        self.assertEqual(
            [row["name"] for row in benchmark["ranked_papers"][:2]],
            ["Clean Evolution Nature", "Blocked Evolution Nature"],
        )
        self.assertGreater(
            benchmark["ranked_papers"][0]["benchmark_score"],
            benchmark["ranked_papers"][1]["benchmark_score"],
        )
        self.assertEqual(
            benchmark["ranked_papers"][1]["self_evolution_status"],
            "blocked",
        )
        self.assertEqual(
            benchmark["ranked_papers"][1]["self_evolution_required_failure_count"],
            2,
        )
        self.assertIn(
            "repair_ownership_gap",
            benchmark["summary"]["top_self_evolution_risks"],
        )
        self.assertEqual(benchmark["summary"]["blocked_self_evolution_count"], 1)
        self.assertEqual(
            benchmark["summary"]["avg_self_evolution_required_failure_count"],
            1.0,
        )
        self.assertIn(
            "self-evolution",
            benchmark["ranked_papers"][1]["recommendation"].lower(),
        )

    def test_build_readiness_benchmark_should_penalize_blocked_process_alignment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_clean_process",
                idea_name="Clean Process Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                process_alignment={
                    "summary": {
                        "overall_score": 90.0,
                        "blocked_process_count": 0,
                        "missing_process_count": 0,
                        "top_process_risks": {},
                    }
                },
                self_evolution={
                    "summary": {
                        "status": "ready",
                        "score": 90.0,
                        "dominant_lane": "section_rewrite",
                        "dominant_role": "clarity",
                    },
                    "self_check": {
                        "status": "ready",
                        "score": 90.0,
                        "required_failures": [],
                    },
                    "stage_risks": [],
                },
            )
            _create_quality_run(
                research_root,
                "paper_blocked_process",
                idea_name="Blocked Process Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=92.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.1,
                claim_alignment_score=4.0,
                numeric_coverage_score=4.0,
                breakthrough_score=4.3,
                evidence_density_score=3.7,
                contribution_count=3,
                process_alignment={
                    "summary": {
                        "overall_score": 58.0,
                        "blocked_process_count": 1,
                        "missing_process_count": 1,
                        "top_process_risks": {"exploration_graph_gap": 1},
                    }
                },
                self_evolution={
                    "summary": {
                        "status": "ready",
                        "score": 90.0,
                        "dominant_lane": "section_rewrite",
                        "dominant_role": "clarity",
                    },
                    "self_check": {
                        "status": "ready",
                        "score": 90.0,
                        "required_failures": [],
                    },
                    "stage_risks": [],
                },
            )

            benchmark = build_readiness_benchmark(
                research_root,
                target_venue="nature",
                top_n=5,
                max_entries=10,
            )

        self.assertEqual(
            [row["name"] for row in benchmark["ranked_papers"][:2]],
            ["Clean Process Nature", "Blocked Process Nature"],
        )
        self.assertGreater(
            benchmark["ranked_papers"][0]["benchmark_score"],
            benchmark["ranked_papers"][1]["benchmark_score"],
        )
        self.assertEqual(
            benchmark["ranked_papers"][1]["process_alignment_blocked_process_count"],
            1,
        )
        self.assertIn(
            "exploration_graph_gap",
            benchmark["summary"]["top_process_alignment_risks"],
        )
        self.assertEqual(
            benchmark["summary"]["avg_process_alignment_blocked_count"],
            0.5,
        )
        self.assertIn(
            "process alignment",
            benchmark["ranked_papers"][1]["recommendation"].lower(),
        )

    def test_export_readiness_benchmark_markdown_should_include_summary_and_recommendations(self) -> None:
        benchmark = {
            "generated_at": "2026-03-15T00:00:00",
            "research_root": "/tmp/research",
            "target_venue": "nature",
            "summary": {
                "entries": 1,
                "venue_match_count": 1,
                "ready_count": 0,
                "gate_pass_count": 0,
                "avg_benchmark_score": 72.5,
                "avg_submission_priority": 68.0,
                "avg_blocker_count": 2.0,
                "avg_fallback_count": 1.0,
                "avg_strict_fallback_count": 0.5,
                "avg_stage_overall_score": 72.0,
                "avg_blocked_stage_count": 1.0,
                "avg_attention_stage_count": 1.0,
                "avg_missing_stage_count": 0.0,
                "avg_process_alignment_score": 64.0,
                "avg_process_alignment_blocked_count": 1.0,
                "avg_process_alignment_missing_count": 1.0,
                "avg_self_evolution_score": 63.0,
                "avg_self_evolution_required_failure_count": 1.0,
                "blocked_self_evolution_count": 1,
                "needs_attention_self_evolution_count": 0,
                "top_blocker_categories": {"breakthrough": 1},
                "top_fallback_kinds": {"heuristic_ranking": 1},
                "top_stage_standard_risks": {"review_evidence_gap": 1},
                "top_process_alignment_risks": {"exploration_graph_gap": 1},
                "top_self_evolution_risks": {"repair_ownership_gap": 1},
                "top_gap_dimensions": {"breakthrough": 1},
            },
            "ranked_papers": [
                {
                    "name": "Beta Nature",
                    "benchmark_score": 72.5,
                    "submission_status": "needs_work",
                    "quality_gate_passed": False,
                    "paper_target_venue": "nature",
                    "venue_match": True,
                    "submission_priority_score": 68.0,
                    "submission_priority_tier": "promising_but_revise",
                    "blocker_count": 2,
                    "fallback_count": 1,
                    "strict_fallback_count": 0,
                    "stage_overall_score": 72.0,
                    "blocked_stage_count": 1,
                    "needs_attention_stage_count": 1,
                    "missing_stage_count": 0,
                    "top_standard_risks": ["review_evidence_gap"],
                    "process_alignment_overall_score": 64.0,
                    "process_alignment_blocked_process_count": 1,
                    "process_alignment_missing_process_count": 1,
                    "top_process_alignment_risks": ["exploration_graph_gap"],
                    "self_evolution_status": "blocked",
                    "self_evolution_score": 63.0,
                    "self_evolution_required_failure_count": 1,
                    "top_self_evolution_risks": ["repair_ownership_gap"],
                    "recommendation": "Close the breakthrough gap first (3.10 < 4.00).",
                    "relative_run_dir": "papers/paper_beta",
                    "failing_metrics": [{"name": "breakthrough", "gap": 0.9}],
                    "top_blockers": [
                        "breakthrough potential below Nature-style bar (3.10 < 4.00)"
                    ],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "reports" / "readiness.md"
            exported = export_readiness_benchmark_markdown(benchmark, output_path)
            content = Path(exported).read_text(encoding="utf-8")

        self.assertEqual(exported, str(output_path))
        self.assertIn("# Readiness Benchmark", content)
        self.assertIn("## Top Blocker Categories", content)
        self.assertIn("## Top Fallback Kinds", content)
        self.assertIn("## Top Stage Standard Risks", content)
        self.assertIn("## Top Process Alignment Risks", content)
        self.assertIn("## Top Self-Evolution Risks", content)
        self.assertIn("### Beta Nature", content)
        self.assertIn("- Fallbacks: 1 (strict=0)", content)
        self.assertIn("- Stage standards: score=72.0 blocked=1 attention=1 missing=0", content)
        self.assertIn("- Process alignment: score=64.0 blocked=1 missing=1", content)
        self.assertIn("- Self-evolution: status=blocked score=63.0 required_failures=1", content)
        self.assertIn("- Top gaps: breakthrough gap=0.9", content)
        self.assertIn("review_evidence_gap", content)
        self.assertIn("exploration_graph_gap", content)
        self.assertIn("repair_ownership_gap", content)
        self.assertIn("Close the breakthrough gap first", content)

    def test_research_manager_should_follow_runtime_output_env_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            research_root = Path(td)
            _create_quality_run(
                research_root,
                "paper_alpha",
                idea_name="Alpha Nature",
                target_venue="nature",
                ready=True,
                gate_passed=True,
                submission_priority=95.0,
                quality_score=4.5,
                rigor_score=4.2,
                claim_support_score=4.2,
                claim_alignment_score=4.1,
                numeric_coverage_score=4.0,
                breakthrough_score=4.4,
                evidence_density_score=3.8,
                contribution_count=3,
            )
            with mock.patch.dict(
                "os.environ",
                {"RESEARCH_OUTPUT_DIR": str(research_root)},
                clear=False,
            ):
                manager = ResearchManager()
                benchmark = manager.readiness_benchmark(target_venue="nature", top_n=3)

        self.assertEqual(manager.research_dir, research_root.resolve())
        self.assertEqual(benchmark["summary"]["entries"], 1)
        self.assertEqual(benchmark["ranked_papers"][0]["name"], "Alpha Nature")


if __name__ == "__main__":
    unittest.main()
