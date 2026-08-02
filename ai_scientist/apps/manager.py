#!/usr/bin/env python3
"""
XScientist research management tool
用于管理 ./research_output 目录中的论文和批次
"""

from __future__ import annotations

import json
import os
import os.path as osp
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])

from ai_scientist.config.paths import resolve_output_path
from ai_scientist.utils.run_index import (
    load_run_index,
    rebuild_run_index,
    run_index_path,
)
from ai_scientist.utils.auth_session import require_login
from ai_scientist.utils.readiness_benchmark import (
    build_readiness_benchmark,
    export_readiness_benchmark_markdown,
)
from ai_scientist.utils.experiment_registry import load_experiment_records
from ai_scientist.utils.pipeline_contracts import (
    iter_project_roots,
    load_contract_artifact,
    load_pipeline_manifest,
)
from ai_scientist.utils.process_alignment import build_process_alignment
from ai_scientist.utils.review_jobs import compute_review_repair_metrics
from ai_scientist.utils.self_evolution import build_self_evolution
from ai_scientist.utils.stage_standards import build_stage_standards
from ai_scientist.apps.manager_ranking import (
    _passes_submission_filters,
    _rewrite_board_sort_key,
    _submission_priority_sort_key,
    _suggest_rewrite_next_step,
)
from ai_scientist.apps.manager_reports import (
    render_repair_board_markdown,
    render_rewrite_board_markdown,
    render_shortlist_markdown,
    render_submission_board_markdown,
)


class ResearchManager:
    """研究管理器"""

    _submission_priority_sort_key = staticmethod(_submission_priority_sort_key)
    _passes_submission_filters = staticmethod(_passes_submission_filters)
    _rewrite_board_sort_key = staticmethod(_rewrite_board_sort_key)
    _suggest_rewrite_next_step = staticmethod(_suggest_rewrite_next_step)

    def __init__(self, research_dir: Optional[str] = None):
        self.research_dir = (
            Path(research_dir).expanduser().resolve()
            if research_dir is not None
            else resolve_output_path().resolve()
        )
        self.batches_dir = self.research_dir / "batches"
        self.papers_dir = self.research_dir / "papers"
        self.ideas_dir = self.research_dir / "ideas"
        self.experiments_dir = self.research_dir / "experiments"

    def list_batches(self) -> List[Dict]:
        """列出所有批次"""
        batches = []
        if not self.batches_dir.exists():
            return batches

        for batch_path in sorted(self.batches_dir.iterdir()):
            if batch_path.is_dir() and batch_path.name.startswith("batch_"):
                progress_file = batch_path / "progress.json"
                progress = {}
                if progress_file.exists():
                    with open(progress_file, "r") as f:
                        progress = json.load(f)

                batches.append(
                    {
                        "name": batch_path.name,
                        "path": str(batch_path),
                        "created_at": datetime.fromtimestamp(
                            batch_path.stat().st_ctime
                        ).isoformat(),
                        "progress": progress,
                    }
                )

        return batches

    def list_papers(
        self, paper_type: str = None, sort_by: str = "modified"
    ) -> List[Dict]:
        """列出所有论文（从独立论文文件夹中）"""
        papers = []

        index_entries = self._get_index_entries(category="papers")
        indexed_dirs = []
        seen_paths = set()
        for entry in index_entries.values():
            run_path = Path(entry["path"])
            if run_path.is_dir() and run_path not in seen_paths:
                indexed_dirs.append(run_path)
                seen_paths.add(run_path)

        if not self.papers_dir.exists():
            paper_dirs = indexed_dirs
        else:
            paper_dirs = indexed_dirs or sorted(
                self.papers_dir.iterdir(),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            )

        # 遍历papers目录下的所有paper_*文件夹
        for paper_folder in paper_dirs:
            if not paper_folder.is_dir() or not paper_folder.name.startswith("paper_"):
                continue

            # 从文件夹名解析论文类型
            # 格式: paper_YYYYMMDD_HHMMSS_idea_name_type
            parts = paper_folder.name.split("_")
            if len(parts) >= 4:
                folder_paper_type = parts[-1]  # 最后一部分是类型

                # 如果指定了类型，进行过滤
                if paper_type and folder_paper_type != paper_type:
                    continue

            # 查找paper.pdf
            pdf_file = paper_folder / "paper.pdf"
            if pdf_file.exists():
                # 尝试读取idea.json获取更多信息
                idea_file = paper_folder / "idea.json"
                idea_name = paper_folder.name
                if idea_file.exists():
                    try:
                        with open(idea_file, "r") as f:
                            idea_data = json.load(f)
                            idea_name = idea_data.get("Name", paper_folder.name)
                    except:
                        pass

                papers.append(
                    {
                        "name": idea_name,
                        "folder": paper_folder.name,
                        "path": str(pdf_file),
                        "type": folder_paper_type if len(parts) >= 4 else "unknown",
                        "size": pdf_file.stat().st_size,
                        "created_at": datetime.fromtimestamp(
                            paper_folder.stat().st_ctime
                        ).isoformat(),
                        "modified_at": datetime.fromtimestamp(
                            paper_folder.stat().st_mtime
                        ).isoformat(),
                        "latest_stage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("latest_stage"),
                        "has_reviews": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("has_reviews"),
                        "batch_name": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("batch_name"),
                        "batch_dir": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("batch_dir"),
                        "daemon_name": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("daemon_name"),
                        "source_name": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("source_name"),
                        "source_key": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("source_key"),
                        "source_type": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("source_type"),
                        "source_value": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("source_value"),
                        "source_target_venue": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("source_target_venue"),
                        "source_paper_types": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("source_paper_types"),
                        "quality_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("quality_score"),
                        "rigor_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rigor_score"),
                        "claim_support_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("claim_support_score"),
                        "claim_alignment_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("claim_alignment_score"),
                        "numeric_coverage_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("numeric_coverage_score"),
                        "breakthrough_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("breakthrough_score"),
                        "claims_detected": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("claims_detected"),
                        "unsupported_claims_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("unsupported_claims_count"),
                        "suggested_claim_rewrites_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("suggested_claim_rewrites_count"),
                        "num_figures": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("num_figures"),
                        "num_tables": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("num_tables"),
                        "evidence_density_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("evidence_density_score"),
                        "key_results_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("key_results_count"),
                        "structured_results_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("structured_results_count"),
                        "contribution_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("contribution_count"),
                        "target_venue": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("target_venue"),
                        "submission_status": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("submission_status"),
                        "submission_package_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("submission_package_file"),
                        "narrative_map_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("narrative_map_file"),
                        "result_story_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("result_story_file"),
                        "contribution_map_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("contribution_map_file"),
                        "editor_pitch_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("editor_pitch_file"),
                        "rebuttal_package_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rebuttal_package_file"),
                        "risk_register_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("risk_register_file"),
                        "cover_letter_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("cover_letter_file"),
                        "abstract_polish_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("abstract_polish_file"),
                        "impact_brief_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("impact_brief_file"),
                        "contribution_bullets_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("contribution_bullets_file"),
                        "strongest_claims_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("strongest_claims_file"),
                        "submission_manifest_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("submission_manifest_file"),
                        "submission_dashboard_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("submission_dashboard_file"),
                        "risk_language_plan_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("risk_language_plan_file"),
                        "claim_softening_plan_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("claim_softening_plan_file"),
                        "rewrite_effectiveness_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_effectiveness_file"),
                        "rewrite_trace_summary_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_trace_summary_file"),
                        "rewrite_round_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_round_count"),
                        "rewrite_priority_gain_total": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_priority_gain_total"),
                        "rewrite_quality_gain_total": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_quality_gain_total"),
                        "rewrite_best_round_priority_delta": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_best_round_priority_delta"),
                        "rewrite_top_frontmatter_style": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_top_frontmatter_style"),
                        "rewrite_top_section_style": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_top_section_style"),
                        "rewrite_top_section": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("rewrite_top_section"),
                        "submission_priority_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("submission_priority_score"),
                        "submission_priority_tier": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("submission_priority_tier"),
                        "fallback_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("fallback_count"),
                        "strict_fallback_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("strict_fallback_count"),
                        "fallback_stage_counts": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("fallback_stage_counts"),
                        "fallback_kind_counts": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("fallback_kind_counts"),
                        "latest_fallback_event": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("latest_fallback_event"),
                        "stage_standards_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("stage_standards_file"),
                        "repair_plan_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_file"),
                        "self_evolution_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_file"),
                        "process_alignment_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_file"),
                        "stage_overall_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("stage_overall_score"),
                        "ready_stage_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("ready_stage_count"),
                        "blocked_stage_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("blocked_stage_count"),
                        "needs_attention_stage_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("needs_attention_stage_count"),
                        "missing_stage_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("missing_stage_count"),
                        "blocked_standard_stages": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("blocked_standard_stages"),
                        "attention_standard_stages": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("attention_standard_stages"),
                        "missing_standard_stages": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("missing_standard_stages"),
                        "top_standard_risks": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("top_standard_risks"),
                        "blocker_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("blocker_count"),
                        "critical_revision_actions_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("critical_revision_actions_count"),
                        "quality_rewrite_applied": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("quality_rewrite_applied"),
                        "quality_gate_passed": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("quality_gate_passed"),
                        "quality_status": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("quality_status"),
                        "self_review_rounds_completed": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_rounds_completed"),
                        "self_review_round_gate_ready": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_round_gate_ready"),
                        "self_review_round_gate_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_round_gate_score"),
                        "self_review_round_gate_reasons": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_round_gate_reasons"),
                        "self_review_round_gate_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_round_gate_file"),
                        "review_active_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_active_issue_count"),
                        "review_resolved_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_resolved_issue_count"),
                        "review_persistent_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_persistent_issue_count"),
                        "review_repair_action_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_action_count"),
                        "review_verification_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_verification_count"),
                        "review_bound_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_bound_issue_count"),
                        "review_unbound_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_unbound_issue_count"),
                        "review_bound_active_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_bound_active_issue_count"),
                        "review_target_binding_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_target_binding_coverage"),
                        "review_active_binding_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_active_binding_coverage"),
                        "review_role_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_role_count"),
                        "review_role_coverage_ratio": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_role_coverage_ratio"),
                        "review_resolution_rate": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_resolution_rate"),
                        "review_verification_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_verification_coverage"),
                        "review_repair_queue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_queue_count"),
                        "review_repair_ready_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_ready_count"),
                        "review_repair_verification_ready_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_verification_ready_count"),
                        "review_repair_targeted_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_targeted_count"),
                        "review_repair_queue_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_queue_coverage"),
                        "review_repair_ready_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_ready_coverage"),
                        "review_repair_verification_ready_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_verification_ready_coverage"),
                        "review_repair_targeted_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("review_repair_targeted_coverage"),
                        "repair_plan_task_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_task_count"),
                        "repair_plan_ready_task_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_ready_task_count"),
                        "repair_plan_blocked_task_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_blocked_task_count"),
                        "repair_plan_verification_ready_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_verification_ready_count"),
                        "repair_plan_lane_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_lane_count"),
                        "repair_plan_ready_rate": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_ready_rate"),
                        "repair_plan_verification_ready_rate": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_verification_ready_rate"),
                        "repair_plan_targeted_rate": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("repair_plan_targeted_rate"),
                        "self_evolution_status": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_status"),
                        "self_evolution_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_score"),
                        "self_evolution_lesson_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_lesson_count"),
                        "self_evolution_required_failure_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_required_failure_count"),
                        "self_evolution_dominant_lane": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_dominant_lane"),
                        "self_evolution_dominant_role": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_dominant_role"),
                        "self_evolution_next_cycle_stages": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_next_cycle_stages"),
                        "self_evolution_top_risks": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_evolution_top_risks"),
                        "process_alignment_overall_score": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_overall_score"),
                        "process_alignment_ready_process_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_ready_process_count"),
                        "process_alignment_blocked_process_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_blocked_process_count"),
                        "process_alignment_attention_process_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_attention_process_count"),
                        "process_alignment_missing_process_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_missing_process_count"),
                        "process_alignment_top_risks": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("process_alignment_top_risks"),
                        "self_review_unresolved_critical": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_unresolved_critical"),
                        "self_review_persistent_issues": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_persistent_issues"),
                        "self_review_high_value_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_high_value_coverage"),
                        "self_review_coverage": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_coverage"),
                        "self_review_focus_issue_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_focus_issue_count"),
                        "self_review_next_focus": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("self_review_next_focus"),
                        "experiment_todo_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_count"),
                        "experiment_todo_p0_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_p0_count"),
                        "experiment_todo_top_action": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_top_action"),
                        "experiment_todo_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_file"),
                        "experiment_todo_closed_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_closed_count"),
                        "experiment_todo_unresolved_count": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_unresolved_count"),
                        "experiment_todo_closure_rate": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_closure_rate"),
                        "experiment_todo_p0_closure_rate": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_p0_closure_rate"),
                        "experiment_todo_progress_file": index_entries.get(
                            self._relative_output_path(paper_folder), {}
                        ).get("experiment_todo_progress_file"),
                    }
                )

        if sort_by == "quality":
            papers.sort(key=self._submission_priority_sort_key, reverse=True)
        else:
            papers.sort(key=lambda paper: paper.get("modified_at", ""), reverse=True)

        return papers

    def list_ideas(self) -> List[Dict]:
        """列出所有想法"""
        ideas = []

        # 检查主ideas目录
        if self.ideas_dir.exists():
            for idea_file in self.ideas_dir.glob("*.json"):
                try:
                    with open(idea_file, "r") as f:
                        idea_data = json.load(f)
                    if isinstance(idea_data, list):
                        for idx, idea in enumerate(idea_data):
                            ideas.append(
                                {
                                    "name": idea.get("Name", f"idea_{idx}"),
                                    "title": idea.get("Title", ""),
                                    "source": f"{idea_file.name}#{idx}",
                                    "file": str(idea_file),
                                }
                            )
                except:
                    ideas.append(
                        {
                            "name": idea_file.stem,
                            "title": "",
                            "source": idea_file.name,
                            "file": str(idea_file),
                        }
                    )

        # 检查批次目录中的想法
        if self.batches_dir.exists():
            for batch_dir in self.batches_dir.iterdir():
                if batch_dir.is_dir():
                    ideas_subdir = batch_dir / "ideas"
                    if ideas_subdir.exists():
                        for idea_file in ideas_subdir.glob("*.json"):
                            try:
                                with open(idea_file, "r") as f:
                                    idea_data = json.load(f)
                                if isinstance(idea_data, list):
                                    for idx, idea in enumerate(idea_data):
                                        ideas.append(
                                            {
                                                "name": idea.get("Name", f"idea_{idx}"),
                                                "title": idea.get("Title", ""),
                                                "source": f"{batch_dir.name}/{idea_file.name}#{idx}",
                                                "file": str(idea_file),
                                            }
                                        )
                            except:
                                pass

        return ideas

    def _iter_pipeline_projects(self) -> List[Path]:
        return iter_project_roots(self.research_dir)

    @staticmethod
    def _load_stage_standards(project_root: Path) -> Dict:
        standards = load_contract_artifact(
            project_root,
            "stage_standards",
            default={},
        )
        if isinstance(standards, dict) and (
            "stage_results" in standards or "overall_score" in standards
        ):
            return standards
        return build_stage_standards(project_root)

    @staticmethod
    def _load_self_evolution(project_root: Path) -> Dict:
        evolution = load_contract_artifact(
            project_root,
            "self_evolution",
            default={},
        )
        if isinstance(evolution, dict) and (
            "summary" in evolution or "self_check" in evolution
        ):
            return evolution
        return build_self_evolution(project_root)

    @staticmethod
    def _load_process_alignment(project_root: Path) -> Dict:
        alignment = load_contract_artifact(
            project_root,
            "process_alignment",
            default={},
        )
        if isinstance(alignment, dict) and (
            "summary" in alignment
            or "process_results" in alignment
            or "reference_summary" in alignment
        ):
            return alignment
        return build_process_alignment(project_root)

    def pipeline_status(self, top_n: int = 20) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            manifest = load_pipeline_manifest(project_root)
            artifacts = manifest.get("artifacts", {}) or {}
            research_plan = (
                load_contract_artifact(
                    project_root,
                    "research_plan",
                    default={},
                )
                or {}
            )
            review_state = (
                load_contract_artifact(
                    project_root,
                    "review_state",
                    default={},
                )
                or {}
            )
            repair_plan = (
                load_contract_artifact(
                    project_root,
                    "repair_plan",
                    default={},
                )
                or {}
            )
            self_evolution = self._load_self_evolution(project_root)
            process_alignment = self._load_process_alignment(project_root)
            fallback_summary = manifest.get("fallback_summary") or {}
            execution_policy = (
                research_plan.get("execution_policy")
                if isinstance(research_plan, dict)
                else {}
            )
            if not isinstance(execution_policy, dict):
                execution_policy = {}
            review_repair_metrics = compute_review_repair_metrics(review_state)
            repair_plan_summary = (
                repair_plan.get("summary")
                if isinstance(repair_plan.get("summary"), dict)
                else {}
            )
            self_evolution_summary = (
                self_evolution.get("summary")
                if isinstance(self_evolution.get("summary"), dict)
                else {}
            )
            self_evolution_self_check = (
                self_evolution.get("self_check")
                if isinstance(self_evolution.get("self_check"), dict)
                else {}
            )
            self_evolution_defaults = (
                self_evolution.get("next_cycle_defaults")
                if isinstance(self_evolution.get("next_cycle_defaults"), dict)
                else {}
            )
            process_alignment_summary = (
                process_alignment.get("summary")
                if isinstance(process_alignment.get("summary"), dict)
                else {}
            )
            stage_standards = self._load_stage_standards(project_root)
            standards_summary = (
                stage_standards.get("summary")
                if isinstance(stage_standards.get("summary"), dict)
                else {}
            )
            budget = (
                research_plan.get("budget") if isinstance(research_plan, dict) else {}
            )
            ready = []
            blocked = []
            missing = []
            failed = []
            stale = []
            warnings = []
            for name, artifact in artifacts.items():
                status = str((artifact or {}).get("status") or "missing")
                if status == "ready":
                    ready.append(name)
                elif status == "blocked":
                    blocked.append(name)
                elif status == "failed":
                    failed.append(name)
                elif status == "stale":
                    stale.append(name)
                else:
                    missing.append(name)
                warnings.extend((artifact or {}).get("warnings") or [])
            rows.append(
                {
                    "project": project_root.name,
                    "project_root": str(project_root),
                    "workflow_mode": manifest.get("workflow_mode"),
                    "workflow_label": manifest.get("workflow_label"),
                    "template_profile": manifest.get("template_profile"),
                    "template_capability": manifest.get("template_capability"),
                    "pipeline_goal": manifest.get("pipeline_goal"),
                    "execution_policy": execution_policy.get("policy_name"),
                    "execution_style": execution_policy.get("execution_style"),
                    "evidence_pressure": execution_policy.get("evidence_pressure"),
                    "budget": budget if isinstance(budget, dict) else {},
                    "acceptance_rule_count": len(
                        execution_policy.get("acceptance_rules") or []
                    ),
                    "fallback_count": int(fallback_summary.get("count") or 0),
                    "strict_fallback_count": int(
                        fallback_summary.get("strict_count") or 0
                    ),
                    "fallback_stage_counts": fallback_summary.get("stage_counts") or {},
                    "fallback_kind_counts": fallback_summary.get("kind_counts") or {},
                    "latest_fallback_event": fallback_summary.get("latest_event") or {},
                    "review_active_issue_count": int(
                        review_repair_metrics.get("active_issue_count") or 0
                    ),
                    "review_resolved_issue_count": int(
                        review_repair_metrics.get("resolved_issue_count") or 0
                    ),
                    "review_persistent_issue_count": int(
                        review_repair_metrics.get("persistent_issue_count") or 0
                    ),
                    "review_repair_action_count": int(
                        review_repair_metrics.get("repair_action_count") or 0
                    ),
                    "review_verification_count": int(
                        review_repair_metrics.get("verification_count") or 0
                    ),
                    "review_bound_issue_count": int(
                        review_repair_metrics.get("bound_issue_count") or 0
                    ),
                    "review_unbound_issue_count": int(
                        review_repair_metrics.get("unbound_issue_count") or 0
                    ),
                    "review_bound_active_issue_count": int(
                        review_repair_metrics.get("bound_active_issue_count") or 0
                    ),
                    "review_target_binding_coverage": float(
                        review_repair_metrics.get("target_binding_coverage") or 0.0
                    ),
                    "review_active_binding_coverage": float(
                        review_repair_metrics.get("active_binding_coverage") or 0.0
                    ),
                    "review_role_count": int(
                        review_repair_metrics.get("role_count") or 0
                    ),
                    "review_role_coverage_ratio": float(
                        review_repair_metrics.get("role_coverage_ratio") or 0.0
                    ),
                    "review_resolution_rate": float(
                        review_repair_metrics.get("resolution_rate") or 0.0
                    ),
                    "review_verification_coverage": float(
                        review_repair_metrics.get("verification_coverage") or 0.0
                    ),
                    "review_repair_queue_count": int(
                        review_repair_metrics.get("repair_queue_count") or 0
                    ),
                    "review_repair_ready_count": int(
                        review_repair_metrics.get("repair_ready_count") or 0
                    ),
                    "review_repair_verification_ready_count": int(
                        review_repair_metrics.get("repair_verification_ready_count")
                        or 0
                    ),
                    "review_repair_targeted_count": int(
                        review_repair_metrics.get("repair_targeted_count") or 0
                    ),
                    "review_repair_queue_coverage": float(
                        review_repair_metrics.get("repair_queue_coverage") or 0.0
                    ),
                    "review_repair_ready_coverage": float(
                        review_repair_metrics.get("repair_ready_coverage") or 0.0
                    ),
                    "review_repair_verification_ready_coverage": float(
                        review_repair_metrics.get("repair_verification_ready_coverage")
                        or 0.0
                    ),
                    "review_repair_targeted_coverage": float(
                        review_repair_metrics.get("repair_targeted_coverage") or 0.0
                    ),
                    "repair_plan_task_count": int(
                        repair_plan_summary.get("task_count") or 0
                    ),
                    "repair_plan_ready_task_count": int(
                        repair_plan_summary.get("ready_task_count") or 0
                    ),
                    "repair_plan_blocked_task_count": int(
                        repair_plan_summary.get("blocked_task_count") or 0
                    ),
                    "repair_plan_lane_count": int(
                        repair_plan_summary.get("lane_count") or 0
                    ),
                    "repair_plan_ready_rate": float(
                        repair_plan_summary.get("ready_rate") or 0.0
                    ),
                    "repair_plan_verification_ready_rate": float(
                        repair_plan_summary.get("verification_ready_rate") or 0.0
                    ),
                    "self_evolution_file": (
                        str(project_root / "self_evolution.json")
                        if (project_root / "self_evolution.json").exists()
                        else None
                    ),
                    "self_evolution_status": str(
                        self_evolution_summary.get("status") or ""
                    ).strip(),
                    "self_evolution_score": float(
                        self_evolution_summary.get("score") or 0.0
                    ),
                    "self_evolution_lesson_count": int(
                        self_evolution_summary.get("lesson_count") or 0
                    ),
                    "self_evolution_required_failure_count": len(
                        self_evolution_self_check.get("required_failures") or []
                    ),
                    "self_evolution_dominant_lane": self_evolution_summary.get(
                        "dominant_lane"
                    ),
                    "self_evolution_dominant_role": self_evolution_summary.get(
                        "dominant_role"
                    ),
                    "self_evolution_next_cycle_stages": sorted(
                        str(name).strip()
                        for name in self_evolution_defaults.keys()
                        if str(name).strip()
                    ),
                    "self_evolution_top_risks": list(
                        self_evolution.get("stage_risks") or []
                    ),
                    "process_alignment_file": (
                        str(project_root / "process_alignment.json")
                        if (project_root / "process_alignment.json").exists()
                        else None
                    ),
                    "process_alignment_overall_score": float(
                        process_alignment_summary.get("overall_score") or 0.0
                    ),
                    "process_alignment_ready_process_count": int(
                        process_alignment_summary.get("ready_process_count") or 0
                    ),
                    "process_alignment_blocked_process_count": int(
                        process_alignment_summary.get("blocked_process_count") or 0
                    ),
                    "process_alignment_attention_process_count": int(
                        process_alignment_summary.get("needs_attention_process_count")
                        or 0
                    ),
                    "process_alignment_missing_process_count": int(
                        process_alignment_summary.get("missing_process_count") or 0
                    ),
                    "process_alignment_top_risks": list(
                        (
                            process_alignment_summary.get("top_process_risks") or {}
                        ).keys()
                    ),
                    "ready_count": len(ready),
                    "artifact_total": len(artifacts),
                    "ready_artifacts": ready,
                    "blocked_artifacts": blocked,
                    "failed_artifacts": failed,
                    "stale_artifacts": stale,
                    "missing_artifacts": missing,
                    "stage_overall_score": float(
                        stage_standards.get("overall_score") or 0.0
                    ),
                    "ready_stage_count": int(
                        stage_standards.get("ready_stage_count") or 0
                    ),
                    "blocked_stage_count": int(
                        stage_standards.get("blocked_stage_count") or 0
                    ),
                    "needs_attention_stage_count": int(
                        stage_standards.get("needs_attention_stage_count") or 0
                    ),
                    "missing_stage_count": int(
                        stage_standards.get("missing_stage_count") or 0
                    ),
                    "blocked_standard_stages": standards_summary.get("blocked_stages")
                    or [],
                    "attention_standard_stages": standards_summary.get(
                        "attention_stages"
                    )
                    or [],
                    "missing_standard_stages": standards_summary.get("missing_stages")
                    or [],
                    "top_standard_risks": standards_summary.get("top_risks") or [],
                    "warnings": sorted(
                        set(str(item) for item in warnings if str(item).strip())
                    ),
                    "modified_at": datetime.fromtimestamp(
                        project_root.stat().st_mtime
                    ).isoformat(),
                }
            )
        rows.sort(
            key=lambda item: (
                -int(item.get("blocked_stage_count") or 0),
                -int(item.get("needs_attention_stage_count") or 0),
                -int(item.get("missing_stage_count") or 0),
                float(item.get("stage_overall_score") or 0.0),
                -int(item.get("review_persistent_issue_count") or 0),
                -int(item.get("review_unbound_issue_count") or 0),
                -int(item.get("process_alignment_blocked_process_count") or 0),
                float(item.get("self_evolution_score") or 0.0),
                float(item.get("process_alignment_overall_score") or 0.0),
                -int(item.get("self_evolution_required_failure_count") or 0),
                float(item.get("review_repair_ready_coverage") or 0.0),
                float(item.get("review_active_binding_coverage") or 0.0),
                float(item.get("review_resolution_rate") or 0.0),
                -int(item.get("fallback_count") or 0),
                -int(item.get("strict_fallback_count") or 0),
                -int(item.get("ready_count") or 0),
                len(item.get("blocked_artifacts") or []),
                len(item.get("failed_artifacts") or []),
                item.get("modified_at") or "",
            ),
            reverse=False,
        )
        return rows[:top_n]

    def evolution_board(
        self,
        *,
        top_n: int = 30,
        status: str | None = None,
    ) -> List[Dict]:
        rows: List[Dict] = []
        index_entries = self._get_index_entries()
        for project_root in self._iter_pipeline_projects():
            rel_path = self._relative_output_path(project_root)
            index_entry = index_entries.get(rel_path, {})
            evolution = self._load_self_evolution(project_root)
            program = load_contract_artifact(
                project_root,
                "evolution_program",
                default={},
            )
            program = program if isinstance(program, dict) else {}
            control = load_contract_artifact(
                project_root,
                "evolution_control",
                default={},
            )
            control = control if isinstance(control, dict) else {}
            epoch = (
                program.get("epoch") if isinstance(program.get("epoch"), dict) else {}
            )
            intents = [
                item
                for item in (program.get("intents") or [])
                if isinstance(item, dict)
            ]
            summary = (
                evolution.get("summary")
                if isinstance(evolution.get("summary"), dict)
                else {}
            )
            self_check = (
                evolution.get("self_check")
                if isinstance(evolution.get("self_check"), dict)
                else {}
            )
            evolution_status = str(summary.get("status") or "").strip()
            if status and evolution_status != str(status):
                continue
            lessons = [
                item
                for item in (evolution.get("lessons") or [])
                if isinstance(item, dict)
            ]
            rows.append(
                {
                    "project": project_root.name,
                    "project_root": str(project_root),
                    "name": index_entry.get("name") or project_root.name,
                    "target_venue": index_entry.get("target_venue"),
                    "workflow_mode": index_entry.get("workflow_mode"),
                    "status": evolution_status,
                    "score": float(summary.get("score") or 0.0),
                    "lesson_count": int(summary.get("lesson_count") or 0),
                    "required_failure_count": len(
                        self_check.get("required_failures") or []
                    ),
                    "dominant_lane": summary.get("dominant_lane"),
                    "dominant_role": summary.get("dominant_role"),
                    "stage_risks": list(evolution.get("stage_risks") or []),
                    "next_cycle_defaults": evolution.get("next_cycle_defaults") or {},
                    "top_lessons": lessons[:3],
                    "evolution_program_id": program.get("program_id"),
                    "epoch_id": epoch.get("epoch_id"),
                    "epoch_index": epoch.get("epoch_index"),
                    "epoch_status": epoch.get("status"),
                    "execution_status": control.get("status"),
                    "should_stop": control.get("should_stop"),
                    "trial_count": int(control.get("trial_count") or 0),
                    "remaining_next_intent_ids": list(
                        control.get("next_intent_ids") or []
                    ),
                    "mechanism_diversity": control.get("mechanism_diversity"),
                    "discovery_efficiency": control.get("discovery_efficiency"),
                    "active_intent_count": len(intents),
                    "intent_component_counts": dict(
                        Counter(
                            str(item.get("component_type") or "unknown")
                            for item in intents
                        )
                    ),
                    "intent_search_mode_counts": dict(
                        Counter(
                            str(item.get("search_mode") or "unknown")
                            for item in intents
                        )
                    ),
                    "evaluator_challenge_count": len(
                        program.get("evaluation_challenges") or []
                    ),
                    "top_intents": intents[:3],
                }
            )
        status_rank = {
            "blocked": 0,
            "needs_attention": 1,
            "ready": 2,
            "": 3,
        }
        rows.sort(
            key=lambda item: (
                status_rank.get(str(item.get("status") or ""), 9),
                float(item.get("score") or 0.0),
                -int(item.get("required_failure_count") or 0),
                -int(item.get("lesson_count") or 0),
                item.get("project") or "",
            )
        )
        return rows[:top_n]

    def stage_standards_board(
        self,
        *,
        top_n: int = 60,
        stage: str | None = None,
        status: str | None = None,
    ) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            standards = self._load_stage_standards(project_root)
            for result in standards.get("stage_results") or []:
                if not isinstance(result, dict):
                    continue
                if stage and str(result.get("stage") or "") != str(stage):
                    continue
                if status and str(result.get("status") or "") != str(status):
                    continue
                rows.append(
                    {
                        "project": project_root.name,
                        "project_root": str(project_root),
                        "stage": result.get("stage"),
                        "artifact": result.get("artifact"),
                        "status": result.get("status"),
                        "score": float(result.get("score") or 0.0),
                        "required_failures": list(
                            result.get("required_failures") or []
                        ),
                        "missing_reason": result.get("missing_reason"),
                        "signals": result.get("signals") or {},
                        "criteria_count": len(result.get("criteria") or []),
                        "passed_criteria_count": sum(
                            1
                            for item in (result.get("criteria") or [])
                            if isinstance(item, dict) and item.get("passed")
                        ),
                    }
                )
        status_rank = {
            "blocked": 0,
            "needs_attention": 1,
            "missing": 2,
            "ready": 3,
        }
        rows.sort(
            key=lambda item: (
                status_rank.get(str(item.get("status") or "ready"), 9),
                float(item.get("score") or 0.0),
                -len(item.get("required_failures") or []),
                item.get("project") or "",
                item.get("stage") or "",
            )
        )
        return rows[:top_n]

    def process_board(
        self,
        *,
        top_n: int = 80,
        process: str | None = None,
        status: str | None = None,
    ) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            alignment = self._load_process_alignment(project_root)
            summary = (
                alignment.get("summary")
                if isinstance(alignment.get("summary"), dict)
                else {}
            )
            for result in alignment.get("process_results") or []:
                if not isinstance(result, dict):
                    continue
                process_name = str(result.get("process") or "")
                process_status = str(result.get("status") or "")
                if process and process_name != str(process):
                    continue
                if status and process_status != str(status):
                    continue
                rows.append(
                    {
                        "project": project_root.name,
                        "project_root": str(project_root),
                        "process": process_name,
                        "label": result.get("label"),
                        "focus": result.get("focus"),
                        "status": process_status,
                        "score": float(result.get("score") or 0.0),
                        "required_failures": list(
                            result.get("required_failures") or []
                        ),
                        "missing_reason": result.get("missing_reason"),
                        "signals": result.get("signals") or {},
                        "risks": list(result.get("risks") or []),
                        "artifacts": list(result.get("artifacts") or []),
                        "references": [
                            str((item or {}).get("name") or "")
                            for item in (result.get("references") or [])
                            if str((item or {}).get("name") or "").strip()
                        ],
                        "criteria_count": len(result.get("criteria") or []),
                        "passed_criteria_count": sum(
                            1
                            for item in (result.get("criteria") or [])
                            if isinstance(item, dict) and item.get("passed")
                        ),
                        "overall_score": float(summary.get("overall_score") or 0.0),
                        "blocked_process_count": int(
                            summary.get("blocked_process_count") or 0
                        ),
                    }
                )
        status_rank = {
            "blocked": 0,
            "needs_attention": 1,
            "missing": 2,
            "ready": 3,
        }
        rows.sort(
            key=lambda item: (
                status_rank.get(str(item.get("status") or "ready"), 9),
                float(item.get("score") or 0.0),
                -len(item.get("required_failures") or []),
                -int(item.get("blocked_process_count") or 0),
                item.get("project") or "",
                item.get("process") or "",
            )
        )
        return rows[:top_n]

    def fallback_board(
        self,
        *,
        top_n: int = 30,
        stage: str | None = None,
    ) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            manifest = load_pipeline_manifest(project_root)
            summary = manifest.get("fallback_summary") or {}
            events = [
                event
                for event in (manifest.get("fallback_events") or [])
                if isinstance(event, dict)
            ]
            if stage:
                events = [
                    event
                    for event in events
                    if str(event.get("stage") or "") == str(stage)
                ]
            if not events:
                continue
            stage_counts = Counter(
                str(event.get("stage") or "unknown") for event in events
            )
            kind_counts = Counter(
                str(event.get("fallback_kind") or "unknown") for event in events
            )
            latest_event = max(
                events,
                key=lambda event: str(event.get("recorded_at") or ""),
            )
            rows.append(
                {
                    "project": project_root.name,
                    "project_root": str(project_root),
                    "workflow_mode": manifest.get("workflow_mode"),
                    "template_profile": manifest.get("template_profile"),
                    "fallback_count": len(events),
                    "strict_fallback_count": sum(
                        bool(event.get("strict")) for event in events
                    ),
                    "stage_counts": dict(stage_counts),
                    "kind_counts": dict(kind_counts),
                    "latest_stage": latest_event.get("stage"),
                    "latest_kind": latest_event.get("fallback_kind"),
                    "latest_reason": latest_event.get("reason"),
                    "latest_recorded_at": latest_event.get("recorded_at"),
                    "latest_metadata": latest_event.get("metadata") or {},
                    "manifest_fallback_count": int(summary.get("count") or 0),
                }
            )
        rows.sort(
            key=lambda item: (
                -int(item.get("fallback_count") or 0),
                -int(item.get("strict_fallback_count") or 0),
                item.get("latest_recorded_at") or "",
            ),
            reverse=False,
        )
        return rows[:top_n]

    def idea_board(
        self,
        *,
        top_n: int = 30,
        status: str | None = None,
    ) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            manifest = load_pipeline_manifest(project_root)
            idea_cards = (
                load_contract_artifact(project_root, "idea_cards", default=[]) or []
            )
            for card in idea_cards:
                if not isinstance(card, dict):
                    continue
                card_status = str(card.get("status") or "unknown")
                if status and card_status != status:
                    continue
                rows.append(
                    {
                        "project": project_root.name,
                        "project_root": str(project_root),
                        "workflow_mode": manifest.get("workflow_mode"),
                        "template_profile": manifest.get("template_profile"),
                        "idea_id": card.get("idea_id"),
                        "name": card.get("name"),
                        "title": card.get("title"),
                        "status": card_status,
                        "target_venue": card.get("target_venue"),
                        "datasets": card.get("candidate_datasets") or [],
                        "metrics": card.get("candidate_metrics") or [],
                        "baselines": card.get("candidate_baselines") or [],
                        "compute_risk": card.get("compute_risk"),
                        "minimum_viable_experiment": card.get(
                            "minimum_viable_experiment"
                        ),
                        "modified_at": datetime.fromtimestamp(
                            project_root.stat().st_mtime
                        ).isoformat(),
                    }
                )
        rows.sort(key=lambda item: item.get("modified_at") or "", reverse=True)
        return rows[:top_n]

    def experiment_board(
        self,
        *,
        top_n: int = 50,
        status: str | None = None,
    ) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            records = load_experiment_records(project_root)
            for record in records:
                if not isinstance(record, dict):
                    continue
                record_status = str(record.get("status") or "unknown")
                if status and record_status != status:
                    continue
                rows.append(
                    {
                        "project": project_root.name,
                        "project_root": str(project_root),
                        "record_id": record.get("record_id"),
                        "task_id": record.get("task_id"),
                        "status": record_status,
                        "dataset": record.get("dataset"),
                        "metric": record.get("metric"),
                        "baseline_ref": record.get("baseline_ref"),
                        "entered_storyline": bool(record.get("entered_storyline")),
                        "workflow_mode": record.get("workflow_mode"),
                        "policy_name": record.get("policy_name"),
                        "budget": record.get("budget") or {},
                        "budget_status": record.get("budget_status"),
                        "acceptance_checks": record.get("acceptance_checks") or [],
                        "result_summary": record.get("result_summary") or {},
                        "error_type": record.get("error_type"),
                        "error_message": record.get("error_message"),
                        "finished_at": record.get("finished_at"),
                    }
                )
        rows.sort(
            key=lambda item: (
                item.get("entered_storyline") is not True,
                item.get("budget_status") == "budget_exhausted",
                item.get("status") != "completed",
                item.get("finished_at") or "",
            ),
            reverse=False,
        )
        return rows[:top_n]

    def figure_board(
        self,
        *,
        top_n: int = 50,
        include_blocked: bool = True,
    ) -> List[Dict]:
        rows = []
        for project_root in self._iter_pipeline_projects():
            spec = load_contract_artifact(project_root, "figure_spec", default={}) or {}
            for figure in spec.get("figures", []) or []:
                if not isinstance(figure, dict):
                    continue
                if not include_blocked and figure.get("status") != "ready":
                    continue
                rows.append(
                    {
                        "project": project_root.name,
                        "project_root": str(project_root),
                        "figure_id": figure.get("figure_id"),
                        "claim_id": figure.get("claim_id"),
                        "status": figure.get("status"),
                        "figure_type": figure.get("figure_type"),
                        "paper_slot": figure.get("paper_slot"),
                        "data_files": figure.get("data_files") or [],
                        "source_records": figure.get("source_records") or [],
                        "blocking_reasons": figure.get("blocking_reasons") or [],
                    }
                )
        rows.sort(
            key=lambda item: (
                item.get("status") != "ready",
                len(item.get("blocking_reasons") or []),
                item.get("project") or "",
            ),
            reverse=False,
        )
        return rows[:top_n]

    def source_board(
        self,
        *,
        top_n: int = 30,
        archetype: str | None = None,
    ) -> List[Dict]:
        entries = self._get_index_entries()
        grouped: Dict[str, List[Dict]] = {}
        for entry in entries.values():
            if not isinstance(entry, dict):
                continue
            source_key = (
                entry.get("source_key")
                or entry.get("source_name")
                or entry.get("source_value")
            )
            if not source_key:
                continue
            source_archetype = str(entry.get("source_archetype") or "unknown")
            if archetype and source_archetype != archetype:
                continue
            grouped.setdefault(str(source_key), []).append(entry)

        rows = []
        for source_key, group in grouped.items():
            quality_scores = [
                float(item.get("quality_score"))
                for item in group
                if isinstance(item.get("quality_score"), (int, float))
            ]
            priority_scores = [
                float(item.get("submission_priority_score"))
                for item in group
                if isinstance(item.get("submission_priority_score"), (int, float))
            ]
            self_evolution_scores = [
                float(item.get("self_evolution_score"))
                for item in group
                if isinstance(item.get("self_evolution_score"), (int, float))
            ]
            self_evolution_required_failures = [
                int(item.get("self_evolution_required_failure_count") or 0)
                for item in group
                if isinstance(item.get("self_evolution_required_failure_count"), int)
                or str(
                    item.get("self_evolution_required_failure_count") or ""
                ).isdigit()
            ]
            fallback_counts = [
                int(item.get("fallback_count") or 0)
                for item in group
                if isinstance(item.get("fallback_count"), int)
                or str(item.get("fallback_count") or "").isdigit()
            ]
            strict_fallback_counts = [
                int(item.get("strict_fallback_count") or 0)
                for item in group
                if isinstance(item.get("strict_fallback_count"), int)
                or str(item.get("strict_fallback_count") or "").isdigit()
            ]
            archetype_counts = Counter(
                str(item.get("source_archetype") or "unknown") for item in group
            )
            batch_profile_counts = Counter(
                str(item.get("source_batch_profile") or "unknown") for item in group
            )
            workflow_counts = Counter(
                str(item.get("source_workflow_mode") or "unknown") for item in group
            )
            target_venue_counts = Counter(
                str(item.get("source_target_venue") or "unknown") for item in group
            )
            self_evolution_status_counts = Counter(
                str(item.get("self_evolution_status") or "unknown") for item in group
            )
            evolution_lane_counts = Counter(
                str(item.get("self_evolution_dominant_lane") or "unknown")
                for item in group
                if str(item.get("self_evolution_dominant_lane") or "").strip()
            )
            evolution_role_counts = Counter(
                str(item.get("self_evolution_dominant_role") or "unknown")
                for item in group
                if str(item.get("self_evolution_dominant_role") or "").strip()
            )
            next_cycle_stage_counts = Counter()
            evolution_risk_counts = Counter()
            fallback_kind_counts = Counter()
            fallback_stage_counts = Counter()
            for item in group:
                for key, value in (item.get("fallback_kind_counts") or {}).items():
                    try:
                        fallback_kind_counts[str(key)] += int(value or 0)
                    except (TypeError, ValueError):
                        continue
                for key, value in (item.get("fallback_stage_counts") or {}).items():
                    try:
                        fallback_stage_counts[str(key)] += int(value or 0)
                    except (TypeError, ValueError):
                        continue
                for key in item.get("self_evolution_next_cycle_stages") or []:
                    label = str(key).strip()
                    if label:
                        next_cycle_stage_counts[label] += 1
                for key in item.get("self_evolution_top_risks") or []:
                    label = str(key).strip()
                    if label:
                        evolution_risk_counts[label] += 1
            latest = max(
                group,
                key=lambda item: str(item.get("updated_at") or ""),
            )
            fallback_run_count = (
                sum(count > 0 for count in fallback_counts) if fallback_counts else 0
            )
            avg_fallback_count = (
                sum(fallback_counts) / len(fallback_counts) if fallback_counts else 0.0
            )
            avg_strict_fallback_count = (
                sum(strict_fallback_counts) / len(strict_fallback_counts)
                if strict_fallback_counts
                else 0.0
            )
            avg_self_evolution_score = (
                sum(self_evolution_scores) / len(self_evolution_scores)
                if self_evolution_scores
                else None
            )
            avg_self_evolution_required_failures = (
                sum(self_evolution_required_failures)
                / len(self_evolution_required_failures)
                if self_evolution_required_failures
                else 0.0
            )
            fallback_free_rate = (
                round(
                    max(0.0, 1.0 - (fallback_run_count / max(len(group), 1))),
                    3,
                )
                if group
                else None
            )
            blocked_self_evolution_run_count = sum(
                str(item.get("self_evolution_status") or "") == "blocked"
                for item in group
            )
            attention_self_evolution_run_count = sum(
                str(item.get("self_evolution_status") or "") == "needs_attention"
                for item in group
            )
            ready_self_evolution_run_count = sum(
                str(item.get("self_evolution_status") or "") == "ready"
                for item in group
            )
            rows.append(
                {
                    "source_key": source_key,
                    "source_name": latest.get("source_name"),
                    "source_type": latest.get("source_type"),
                    "source_value": latest.get("source_value"),
                    "run_count": len(group),
                    "ready_count": sum(
                        str(item.get("submission_status") or "") == "ready"
                        for item in group
                    ),
                    "gate_pass_count": sum(
                        item.get("quality_gate_passed") is True for item in group
                    ),
                    "avg_quality_score": (
                        sum(quality_scores) / len(quality_scores)
                        if quality_scores
                        else None
                    ),
                    "avg_submission_priority": (
                        sum(priority_scores) / len(priority_scores)
                        if priority_scores
                        else None
                    ),
                    "avg_self_evolution_score": avg_self_evolution_score,
                    "avg_self_evolution_required_failures": avg_self_evolution_required_failures,
                    "blocked_self_evolution_run_count": blocked_self_evolution_run_count,
                    "needs_attention_self_evolution_run_count": attention_self_evolution_run_count,
                    "ready_self_evolution_run_count": ready_self_evolution_run_count,
                    "avg_fallback_count": avg_fallback_count,
                    "avg_strict_fallback_count": avg_strict_fallback_count,
                    "fallback_run_count": fallback_run_count,
                    "fallback_free_rate": fallback_free_rate,
                    "fallback_kind_counts": dict(fallback_kind_counts),
                    "fallback_stage_counts": dict(fallback_stage_counts),
                    "self_evolution_status_counts": dict(self_evolution_status_counts),
                    "self_evolution_lane_counts": dict(evolution_lane_counts),
                    "self_evolution_role_counts": dict(evolution_role_counts),
                    "self_evolution_next_cycle_stage_counts": dict(
                        next_cycle_stage_counts
                    ),
                    "self_evolution_risk_counts": dict(evolution_risk_counts),
                    "source_archetype": archetype_counts.most_common(1)[0][0],
                    "source_batch_profile": batch_profile_counts.most_common(1)[0][0],
                    "source_workflow_mode": workflow_counts.most_common(1)[0][0],
                    "dominant_self_evolution_lane": (
                        evolution_lane_counts.most_common(1)[0][0]
                        if evolution_lane_counts
                        else None
                    ),
                    "dominant_self_evolution_role": (
                        evolution_role_counts.most_common(1)[0][0]
                        if evolution_role_counts
                        else None
                    ),
                    "target_venue": target_venue_counts.most_common(1)[0][0],
                    "archetype_counts": dict(archetype_counts),
                    "batch_profile_counts": dict(batch_profile_counts),
                    "workflow_mode_counts": dict(workflow_counts),
                    "latest_project": latest.get("project")
                    or latest.get("relative_path"),
                    "latest_path": latest.get("path"),
                    "updated_at": latest.get("updated_at"),
                }
            )
        rows.sort(
            key=lambda item: (
                -int(item.get("ready_count") or 0),
                -int(item.get("gate_pass_count") or 0),
                int(item.get("blocked_self_evolution_run_count") or 0),
                -(float(item.get("avg_self_evolution_score") or 0.0)),
                float(item.get("avg_strict_fallback_count") or 0.0),
                float(item.get("avg_fallback_count") or 0.0),
                -float(item.get("avg_submission_priority") or -1),
                item.get("updated_at") or "",
            ),
            reverse=False,
        )
        return rows[:top_n]

    def source_mix_advisory(
        self,
        *,
        desired_policy: str | None = None,
        top_n: int = 50,
    ) -> Dict:
        rows = self.source_board(top_n=top_n)
        if not rows:
            return {
                "desired_policy": desired_policy,
                "summary": {
                    "source_count": 0,
                    "archetype_counts": {},
                    "workflow_mode_counts": {},
                    "batch_profile_counts": {},
                    "dominant_archetype": None,
                    "dominant_workflow_mode": None,
                },
                "top_sources": [],
                "recommendations": [],
            }

        archetype_counts = Counter(
            str(row.get("source_archetype") or "unknown") for row in rows
        )
        workflow_counts = Counter(
            str(row.get("source_workflow_mode") or "unknown") for row in rows
        )
        batch_profile_counts = Counter(
            str(row.get("source_batch_profile") or "unknown") for row in rows
        )
        evolution_lane_counts = Counter()
        evolution_risk_counts = Counter()
        blocked_self_evolution_run_count = 0
        attention_self_evolution_run_count = 0
        self_evolution_scores: list[float] = []
        for row in rows:
            blocked_self_evolution_run_count += int(
                row.get("blocked_self_evolution_run_count") or 0
            )
            attention_self_evolution_run_count += int(
                row.get("needs_attention_self_evolution_run_count") or 0
            )
            if isinstance(row.get("avg_self_evolution_score"), (int, float)):
                self_evolution_scores.append(float(row.get("avg_self_evolution_score")))
            for key, value in (row.get("self_evolution_lane_counts") or {}).items():
                evolution_lane_counts[str(key)] += int(value or 0)
            for key, value in (row.get("self_evolution_risk_counts") or {}).items():
                evolution_risk_counts[str(key)] += int(value or 0)
        desired_to_archetype = {
            "classic_pipeline": "template_first",
            "agentic_tree": "frontier_exploration",
            "program_driven": "program_guarded",
            "writing_studio": "writing_polish",
            "review_board": "review_hardening",
            "multi_agent_board": "paper_hardening_board",
        }
        recommendations = []
        top_sources = rows[:5]

        aligned_sources = []
        if desired_policy:
            aligned_sources = [
                row
                for row in rows
                if row.get("source_workflow_mode") == desired_policy
                or row.get("source_archetype")
                == desired_to_archetype.get(str(desired_policy))
            ]
            if not aligned_sources:
                recommendations.append(
                    {
                        "tier": "diversify",
                        "label": "missing_desired_policy_source",
                        "recommendation": (
                            f"No source currently aligns with desired policy {desired_policy}; "
                            "add or repurpose a source for that research posture."
                        ),
                    }
                )

        strongest = next(
            (
                row
                for row in rows
                if (
                    (row.get("ready_count") or 0) > 0
                    or (row.get("gate_pass_count") or 0) > 0
                    or (row.get("avg_submission_priority") or 0) >= 85
                )
                and float(row.get("avg_strict_fallback_count") or 0.0) <= 0.5
            ),
            None,
        )
        if strongest is not None:
            recommendations.append(
                {
                    "tier": "promote",
                    "label": "promote_top_source",
                    "source": strongest.get("source_name")
                    or strongest.get("source_key"),
                    "recommendation": (
                        f"Lean harder on {strongest.get('source_name') or strongest.get('source_key')}: "
                        f"it leads the mix with ready={strongest.get('ready_count')} gate={strongest.get('gate_pass_count')} "
                        f"and avg priority={strongest.get('avg_submission_priority')}."
                    ),
                }
            )
        elif rows:
            observed = rows[0]
            recommendations.append(
                {
                    "tier": "observe",
                    "label": "observe_top_source",
                    "source": observed.get("source_name") or observed.get("source_key"),
                    "recommendation": (
                        f"Keep observing {observed.get('source_name') or observed.get('source_key')}: "
                        "it currently leads the mix, but more cycles are needed before strong promotion or deprioritization."
                    ),
                }
            )

        evolution_ready_sources = [
            row
            for row in rows
            if float(row.get("avg_self_evolution_score") or 0.0) >= 85.0
            and int(row.get("blocked_self_evolution_run_count") or 0) == 0
        ]
        if evolution_ready_sources:
            best_evolution_source = sorted(
                evolution_ready_sources,
                key=lambda row: (
                    -float(row.get("avg_self_evolution_score") or 0.0),
                    -int(row.get("ready_count") or 0),
                    str(row.get("source_name") or row.get("source_key") or ""),
                ),
            )[0]
            recommendations.append(
                {
                    "tier": "promote",
                    "label": "promote_evolution_ready_source",
                    "source": best_evolution_source.get("source_name")
                    or best_evolution_source.get("source_key"),
                    "recommendation": (
                        f"Promote {best_evolution_source.get('source_name') or best_evolution_source.get('source_key')}: "
                        f"it combines strong output quality with avg self-evolution score "
                        f"{round(float(best_evolution_source.get('avg_self_evolution_score') or 0.0), 1)} "
                        "and no blocked self-evolution runs."
                    ),
                }
            )

        weak_sources = [
            row
            for row in rows
            if (row.get("run_count") or 0) >= 3
            and (row.get("ready_count") or 0) == 0
            and (row.get("gate_pass_count") or 0) == 0
        ]
        if weak_sources:
            weakest = sorted(
                weak_sources,
                key=lambda row: (
                    float(row.get("avg_submission_priority") or -1),
                    -(row.get("run_count") or 0),
                ),
            )[0]
            recommendations.append(
                {
                    "tier": "deprioritize",
                    "label": "deprioritize_weak_source",
                    "source": weakest.get("source_name") or weakest.get("source_key"),
                    "recommendation": (
                        f"Reduce cycles for {weakest.get('source_name') or weakest.get('source_key')}: "
                        f"it has run {weakest.get('run_count')} times without ready or gate-passed outcomes."
                    ),
                }
            )

        fallback_heavy_sources = [
            row
            for row in rows
            if (row.get("run_count") or 0) >= 1
            and (
                float(row.get("avg_strict_fallback_count") or 0.0) >= 1.0
                or float(row.get("avg_fallback_count") or 0.0) >= 2.0
            )
        ]
        if fallback_heavy_sources:
            heaviest = sorted(
                fallback_heavy_sources,
                key=lambda row: (
                    -float(row.get("avg_strict_fallback_count") or 0.0),
                    -float(row.get("avg_fallback_count") or 0.0),
                    row.get("source_name") or row.get("source_key") or "",
                ),
            )[0]
            recommendations.append(
                {
                    "tier": "quality",
                    "label": "reduce_fallback_debt",
                    "source": heaviest.get("source_name") or heaviest.get("source_key"),
                    "recommendation": (
                        f"Reduce fallback debt for {heaviest.get('source_name') or heaviest.get('source_key')}: "
                        f"avg strict fallback={round(float(heaviest.get('avg_strict_fallback_count') or 0.0), 2)}, "
                        f"avg fallback={round(float(heaviest.get('avg_fallback_count') or 0.0), 2)}."
                    ),
                }
            )

        blocked_evolution_sources = [
            row
            for row in rows
            if int(row.get("blocked_self_evolution_run_count") or 0) >= 1
        ]
        if blocked_evolution_sources:
            weakest_evolution = sorted(
                blocked_evolution_sources,
                key=lambda row: (
                    -int(row.get("blocked_self_evolution_run_count") or 0),
                    -float(row.get("avg_self_evolution_required_failures") or 0.0),
                    str(row.get("source_name") or row.get("source_key") or ""),
                ),
            )[0]
            recommendations.append(
                {
                    "tier": "repair",
                    "label": "repair_self_evolution_debt",
                    "source": weakest_evolution.get("source_name")
                    or weakest_evolution.get("source_key"),
                    "recommendation": (
                        f"Reduce self-evolution debt for {weakest_evolution.get('source_name') or weakest_evolution.get('source_key')}: "
                        f"blocked self-evolution runs={weakest_evolution.get('blocked_self_evolution_run_count')}, "
                        f"avg required failures={round(float(weakest_evolution.get('avg_self_evolution_required_failures') or 0.0), 2)}. "
                        "Use a program-revision or reviewer-hardening lane before scaling this source."
                    ),
                }
            )

        dominant_archetype, dominant_archetype_count = archetype_counts.most_common(1)[
            0
        ]
        if len(archetype_counts) <= 2 and dominant_archetype_count >= max(
            3, len(rows) - 1
        ):
            recommendations.append(
                {
                    "tier": "rebalance",
                    "label": "mix_too_narrow",
                    "recommendation": (
                        f"The source mix is narrow and dominated by {dominant_archetype}; "
                        "add a complementary archetype to avoid overfitting the research loop."
                    ),
                }
            )

        return {
            "desired_policy": desired_policy,
            "summary": {
                "source_count": len(rows),
                "archetype_counts": dict(archetype_counts),
                "workflow_mode_counts": dict(workflow_counts),
                "batch_profile_counts": dict(batch_profile_counts),
                "dominant_archetype": dominant_archetype,
                "dominant_workflow_mode": workflow_counts.most_common(1)[0][0],
                "avg_self_evolution_score": (
                    round(sum(self_evolution_scores) / len(self_evolution_scores), 2)
                    if self_evolution_scores
                    else None
                ),
                "blocked_self_evolution_run_count": blocked_self_evolution_run_count,
                "needs_attention_self_evolution_run_count": attention_self_evolution_run_count,
                "dominant_self_evolution_lane": (
                    evolution_lane_counts.most_common(1)[0][0]
                    if evolution_lane_counts
                    else None
                ),
                "top_self_evolution_risks": dict(evolution_risk_counts.most_common(6)),
            },
            "top_sources": top_sources,
            "recommendations": recommendations[:5],
        }

    def _score_source_batch_candidate(
        self,
        row: Dict,
        *,
        desired_policy: str | None = None,
    ) -> float:
        desired_to_archetype = {
            "classic_pipeline": "template_first",
            "agentic_tree": "frontier_exploration",
            "program_driven": "program_guarded",
            "writing_studio": "writing_polish",
            "review_board": "review_hardening",
            "multi_agent_board": "paper_hardening_board",
        }
        score = float(row.get("run_count") or 0)
        score += float(row.get("ready_count") or 0) * 6.0
        score += float(row.get("gate_pass_count") or 0) * 8.0
        if isinstance(row.get("avg_submission_priority"), (int, float)):
            score += float(row["avg_submission_priority"]) / 10.0
        if isinstance(row.get("avg_quality_score"), (int, float)):
            score += float(row["avg_quality_score"]) * 2.0
        if isinstance(row.get("avg_self_evolution_score"), (int, float)):
            score += float(row["avg_self_evolution_score"]) / 12.0
        score -= float(row.get("avg_self_evolution_required_failures") or 0.0) * 3.0
        score -= float(row.get("blocked_self_evolution_run_count") or 0) * 4.0
        score -= float(row.get("avg_fallback_count") or 0.0) * 2.5
        score -= float(row.get("avg_strict_fallback_count") or 0.0) * 5.0
        if desired_policy:
            if row.get("source_workflow_mode") == desired_policy:
                score += 10.0
            if row.get("source_archetype") == desired_to_archetype.get(
                str(desired_policy)
            ):
                score += 5.0
        return round(score, 2)

    def source_next_batch_advisory(
        self,
        *,
        desired_policy: str | None = None,
        top_n: int = 50,
        max_slots: int = 3,
    ) -> Dict:
        rows = self.source_board(top_n=top_n)
        mix = self.source_mix_advisory(
            desired_policy=desired_policy,
            top_n=top_n,
        )
        if not rows:
            return {
                "desired_policy": desired_policy,
                "summary": mix.get("summary") or {},
                "cadence": {
                    "label": "no_sources",
                    "reason": "No source lineage exists yet, so there is nothing to orchestrate.",
                },
                "slots": [],
                "recommendations": mix.get("recommendations") or [],
            }

        recommendation_labels = {
            str(item.get("label")): item
            for item in (mix.get("recommendations") or [])
            if isinstance(item, dict) and item.get("label")
        }
        dominant_archetype = str(
            ((mix.get("summary") or {}).get("dominant_archetype") or "")
        ).strip()
        dominant_workflow = str(
            ((mix.get("summary") or {}).get("dominant_workflow_mode") or "")
        ).strip()
        hardening_archetypes = {
            "program_guarded",
            "writing_polish",
            "review_hardening",
        }
        desired_to_archetype = {
            "classic_pipeline": "template_first",
            "agentic_tree": "frontier_exploration",
            "program_driven": "program_guarded",
            "writing_studio": "writing_polish",
            "review_board": "review_hardening",
            "multi_agent_board": "paper_hardening_board",
        }
        used_keys: set[str] = set()
        slots: List[Dict] = []

        def row_key(row: Dict) -> str:
            return str(row.get("source_key") or row.get("source_name") or "")

        def pick_best(candidates: List[Dict]) -> Dict | None:
            ranked = sorted(
                [row for row in candidates if row_key(row)],
                key=lambda row: (
                    -self._score_source_batch_candidate(
                        row,
                        desired_policy=desired_policy,
                    ),
                    -(row.get("gate_pass_count") or 0),
                    -(row.get("ready_count") or 0),
                    -(row.get("run_count") or 0),
                    str(row.get("source_name") or row.get("source_key") or ""),
                ),
            )
            return ranked[0] if ranked else None

        def add_slot(
            lane: str,
            share: float,
            row: Dict,
            rationale: str,
            focus: str,
        ) -> None:
            key = row_key(row)
            if not key or key in used_keys:
                return
            used_keys.add(key)
            slots.append(
                {
                    "lane": lane,
                    "share": round(share, 2),
                    "source": row.get("source_name") or row.get("source_key"),
                    "source_key": row.get("source_key"),
                    "source_type": row.get("source_type"),
                    "source_value": row.get("source_value"),
                    "source_archetype": row.get("source_archetype"),
                    "source_workflow_mode": row.get("source_workflow_mode"),
                    "source_batch_profile": row.get("source_batch_profile"),
                    "target_venue": row.get("target_venue"),
                    "run_count": row.get("run_count"),
                    "ready_count": row.get("ready_count"),
                    "gate_pass_count": row.get("gate_pass_count"),
                    "avg_quality_score": row.get("avg_quality_score"),
                    "avg_submission_priority": row.get("avg_submission_priority"),
                    "avg_self_evolution_score": row.get("avg_self_evolution_score"),
                    "blocked_self_evolution_run_count": row.get(
                        "blocked_self_evolution_run_count"
                    ),
                    "dominant_self_evolution_lane": row.get(
                        "dominant_self_evolution_lane"
                    ),
                    "top_self_evolution_risks": dict(
                        list((row.get("self_evolution_risk_counts") or {}).items())[:4]
                    ),
                    "batch_score": self._score_source_batch_candidate(
                        row,
                        desired_policy=desired_policy,
                    ),
                    "focus": focus,
                    "rationale": rationale,
                }
            )

        primary = pick_best(rows)
        if primary is not None:
            primary_name = primary.get("source_name") or primary.get("source_key")
            primary_policy = primary.get("source_workflow_mode")
            primary_rationale = (
                f"{primary_name} currently leads with ready={primary.get('ready_count')} "
                f"gate={primary.get('gate_pass_count')} and avg priority={primary.get('avg_submission_priority')}."
            )
            if isinstance(primary.get("avg_self_evolution_score"), (int, float)):
                primary_rationale += (
                    f" Its avg self-evolution score is "
                    f"{round(float(primary.get('avg_self_evolution_score') or 0.0), 1)}."
                )
            if desired_policy and primary_policy == desired_policy:
                primary_rationale += (
                    f" It already aligns with the active {desired_policy} policy."
                )
            add_slot(
                "primary_lane",
                0.5,
                primary,
                primary_rationale,
                "Drive the next batch with the strongest available research posture.",
            )

        unused_rows = [row for row in rows if row_key(row) not in used_keys]
        diversify_candidates = [
            row
            for row in unused_rows
            if row.get("source_archetype") != (primary or {}).get("source_archetype")
        ]
        diversify_non_hardening = [
            row
            for row in diversify_candidates
            if row.get("source_archetype") not in hardening_archetypes
        ]
        diversify = (
            pick_best(diversify_non_hardening)
            or pick_best(diversify_candidates)
            or pick_best(unused_rows)
        )
        if diversify is not None and len(slots) < max_slots:
            diversify_reason = "Keep a second research posture warm so the next batch is not overfit to one source."
            if "mix_too_narrow" in recommendation_labels and dominant_archetype:
                diversify_reason = f"The current portfolio is dominated by {dominant_archetype}, so this lane widens the batch mix."
            elif desired_policy and (
                diversify.get("source_workflow_mode") == desired_policy
                or diversify.get("source_archetype")
                == desired_to_archetype.get(str(desired_policy))
            ):
                diversify_reason = f"This lane improves coverage for the active {desired_policy} policy."
            add_slot(
                "diversification_lane",
                0.3,
                diversify,
                diversify_reason,
                "Preserve exploration breadth or repair a missing workflow posture.",
            )

        unused_rows = [row for row in rows if row_key(row) not in used_keys]
        hardening_candidates = [
            row
            for row in unused_rows
            if row.get("source_archetype") in hardening_archetypes
        ]
        hardening = pick_best(hardening_candidates) or pick_best(unused_rows)
        if hardening is not None and len(slots) < max_slots:
            hardening_reason = "Reserve one lane for evidence packaging, review hardening, or submission-grade convergence."
            if hardening.get("source_archetype") == "review_hardening":
                hardening_reason = "This lane raises reviewer-facing pressure before the next submission push."
            elif hardening.get("source_archetype") == "writing_polish":
                hardening_reason = "This lane focuses the batch on figures, captions, and writing polish."
            elif hardening.get("source_archetype") == "program_guarded":
                hardening_reason = "This lane keeps the batch tied to a budgeted, submission-oriented program."
            if int(hardening.get("blocked_self_evolution_run_count") or 0) > 0:
                hardening_reason += " It also helps absorb self-evolution debt before more open exploration."
            add_slot(
                "hardening_lane",
                0.2,
                hardening,
                hardening_reason,
                "Ensure at least one lane converges toward submission-grade artifacts.",
            )

        cadence_label = "balanced_rotation"
        cadence_reason = "Keep a balanced portfolio so the next batch can discover, validate, and polish in parallel."
        if "missing_desired_policy_source" in recommendation_labels:
            cadence_label = "portfolio_rebalance"
            cadence_reason = f"The active {desired_policy} policy is underrepresented, so the next batch should rebalance the source portfolio before scaling throughput."
        elif "mix_too_narrow" in recommendation_labels:
            cadence_label = "rebalance_then_converge"
            cadence_reason = f"The mix is currently concentrated in {dominant_archetype}; use the next batch to widen the portfolio before converging."
        elif desired_policy == "agentic_tree":
            cadence_label = "explore_then_converge"
            cadence_reason = "Favor a wider first lane, then converge with one harder evidence or review lane."
        elif desired_policy in {
            "program_driven",
            "review_board",
            "writing_studio",
            "multi_agent_board",
        }:
            cadence_label = "submission_hardening_loop"
            cadence_reason = f"Treat the next batch as a {desired_policy} convergence pass with tighter evidence and review discipline."
        elif desired_policy == "classic_pipeline":
            cadence_label = "repeatable_throughput"
            cadence_reason = "Favor stable, repeatable sources while keeping one smaller secondary lane alive."

        return {
            "desired_policy": desired_policy,
            "summary": {
                **(mix.get("summary") or {}),
                "slot_count": len(slots),
                "dominant_archetype": dominant_archetype or None,
                "dominant_workflow_mode": dominant_workflow or None,
            },
            "cadence": {
                "label": cadence_label,
                "reason": cadence_reason,
            },
            "slots": slots,
            "recommendations": mix.get("recommendations") or [],
        }

    def benchmark_trends(
        self,
        *,
        target_venue: str = "nature",
        max_entries: int = 200,
    ) -> Dict:
        benchmark = self.readiness_benchmark(
            target_venue=target_venue,
            max_entries=max_entries,
            top_n=max_entries,
            include_other_venues=True,
        )
        by_day: Dict[str, Dict] = {}
        for row in benchmark.get("all_papers") or []:
            modified_at = str(row.get("modified_at") or "")
            bucket = modified_at[:10] if len(modified_at) >= 10 else "unknown"
            entry = by_day.setdefault(
                bucket,
                {
                    "date": bucket,
                    "count": 0,
                    "ready_count": 0,
                    "gate_pass_count": 0,
                    "benchmark_scores": [],
                    "priority_scores": [],
                },
            )
            entry["count"] += 1
            if row.get("submission_status") == "ready":
                entry["ready_count"] += 1
            if row.get("quality_gate_passed") is True:
                entry["gate_pass_count"] += 1
            if isinstance(row.get("benchmark_score"), (int, float)):
                entry["benchmark_scores"].append(float(row["benchmark_score"]))
            if isinstance(row.get("submission_priority_score"), (int, float)):
                entry["priority_scores"].append(float(row["submission_priority_score"]))

        timeline = []
        for bucket, payload in sorted(by_day.items(), reverse=True):
            scores = payload.pop("benchmark_scores")
            priorities = payload.pop("priority_scores")
            payload["avg_benchmark_score"] = (
                round(sum(scores) / len(scores), 2) if scores else None
            )
            payload["avg_submission_priority"] = (
                round(sum(priorities) / len(priorities), 2) if priorities else None
            )
            timeline.append(payload)
        return {
            "target_venue": target_venue,
            "summary": benchmark.get("summary") or {},
            "timeline": timeline,
        }

    def get_batch_summary(self, batch_name: str) -> Dict:
        """获取批次摘要"""
        batch_path = self.batches_dir / f"batch_{batch_name}"
        if not batch_path.exists():
            return None

        progress_file = batch_path / "progress.json"
        progress = {}
        if progress_file.exists():
            with open(progress_file, "r") as f:
                progress = json.load(f)

        report_file = batch_path / "final_report.json"
        report = {}
        if report_file.exists():
            with open(report_file, "r") as f:
                report = json.load(f)

        # 统计论文
        papers = []
        papers_dir = batch_path / "papers"
        if papers_dir.exists():
            for paper_type in ["icbinb", "normal", "journal", "extended"]:
                type_dir = papers_dir / paper_type
                if type_dir.exists():
                    for pdf_file in type_dir.glob("*.pdf"):
                        papers.append(
                            {
                                "name": pdf_file.name,
                                "type": paper_type,
                                "path": str(pdf_file),
                            }
                        )

        return {
            "batch_name": batch_name,
            "path": str(batch_path),
            "progress": progress,
            "report": report,
            "papers": papers,
            "failure_summary": self._summarize_failures(
                progress.get("papers_failed", [])
            ),
        }

    def _relative_output_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.research_dir.resolve()))
        except ValueError:
            return str(path.resolve())

    def _get_index_entries(self, category: str = None) -> Dict[str, Dict]:
        index = load_run_index(self.research_dir)
        entries = index.get("entries", {})
        if category is None:
            return entries
        return {
            rel_path: entry
            for rel_path, entry in entries.items()
            if entry.get("category") == category
        }

    def _summarize_failures(self, failures: List[Dict]) -> Dict:
        if not failures:
            return {"total": 0, "by_stage": {}, "samples": []}

        stage_counts = Counter()
        samples = []
        for failure in failures:
            stage = failure.get("stage") or "unknown"
            stage_counts[stage] += 1
            if len(samples) < 5:
                samples.append(
                    {
                        "idea_idx": failure.get("idea_idx"),
                        "paper_type": failure.get("paper_type"),
                        "stage": stage,
                        "error": failure.get("error", "")[:200],
                    }
                )

        return {
            "total": len(failures),
            "by_stage": dict(stage_counts),
            "samples": samples,
        }

    def cleanup_old_files(self, days: int = 30, dry_run: bool = True):
        """清理旧文件"""
        import time
        from datetime import timedelta

        cutoff_time = time.time() - (days * 86400)
        files_to_remove = []

        # 检查实验目录
        if self.experiments_dir.exists():
            for exp_dir in self.experiments_dir.iterdir():
                if exp_dir.is_dir() and exp_dir.stat().st_mtime < cutoff_time:
                    files_to_remove.append(("experiment", str(exp_dir)))

        # 检查批次目录
        if self.batches_dir.exists():
            for batch_dir in self.batches_dir.iterdir():
                if batch_dir.is_dir() and batch_dir.stat().st_mtime < cutoff_time:
                    files_to_remove.append(("batch", str(batch_dir)))

        if dry_run:
            print(f"将会删除以下 {days} 天前的文件:")
            for file_type, path in files_to_remove:
                print(f"  [{file_type}] {path}")
        else:
            for file_type, path in files_to_remove:
                import shutil

                try:
                    if osp.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    print(f"已删除: {path}")
                except Exception as e:
                    print(f"删除失败 {path}: {e}")

    def search_papers(self, query: str, paper_type: str = None) -> List[Dict]:
        """搜索论文"""
        papers = self.list_papers(paper_type)
        query_lower = query.lower()

        results = []
        for paper in papers:
            # 在论文名称中搜索
            if query_lower in paper["name"].lower():
                results.append(paper)
                continue

            # 在文件夹内容中搜索（idea.json等）
            paper_folder = Path(paper["path"]).parent
            for json_file in paper_folder.glob("*.json"):
                try:
                    with open(json_file, "r") as f:
                        content = f.read()
                        if query_lower in content.lower():
                            results.append(paper)
                            break
                except:
                    pass

        return results

    def get_paper_details(self, paper_folder: str) -> Dict:
        """获取论文详细信息"""
        paper_path = self.papers_dir / paper_folder
        if not paper_path.exists():
            return None

        details = {
            "folder": paper_folder,
            "path": str(paper_path),
            "files": [],
            "idea": None,
            "reviews": [],
            "quality": None,
        }

        # 列出所有文件
        for item in paper_path.iterdir():
            if item.is_file():
                details["files"].append(
                    {
                        "name": item.name,
                        "size": item.stat().st_size,
                        "type": item.suffix,
                    }
                )

        # 读取想法
        idea_file = paper_path / "idea.json"
        if idea_file.exists():
            try:
                with open(idea_file, "r") as f:
                    details["idea"] = json.load(f)
            except:
                pass

        # 读取审查
        reviews_dir = paper_path / "reviews"
        if reviews_dir.exists():
            for round_dir in sorted(reviews_dir.iterdir()):
                if round_dir.is_dir():
                    review_data = {}
                    for json_file in round_dir.glob("*.json"):
                        try:
                            with open(json_file, "r") as f:
                                review_data[json_file.stem] = json.load(f)
                        except:
                            pass
                    details["reviews"].append(
                        {"round": round_dir.name, "data": review_data}
                    )

        quality_file = paper_path / "quality" / "high_quality_result.json"
        if quality_file.exists():
            try:
                with open(quality_file, "r") as f:
                    details["quality"] = json.load(f)
            except Exception:
                pass

        editor_pitch_file = paper_path / "quality" / "editor_pitch.md"
        if editor_pitch_file.exists():
            try:
                details["editor_pitch"] = editor_pitch_file.read_text(encoding="utf-8")
            except Exception:
                pass

        impact_brief_file = paper_path / "quality" / "impact_brief.md"
        if impact_brief_file.exists():
            try:
                details["impact_brief"] = impact_brief_file.read_text(encoding="utf-8")
            except Exception:
                pass

        contribution_bullets_file = paper_path / "quality" / "contribution_bullets.md"
        if contribution_bullets_file.exists():
            try:
                details["contribution_bullets"] = contribution_bullets_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        strongest_claims_file = paper_path / "quality" / "strongest_claims.md"
        if strongest_claims_file.exists():
            try:
                details["strongest_claims"] = strongest_claims_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        claim_alignment_file = paper_path / "quality" / "claim_alignment_final.json"
        if claim_alignment_file.exists():
            try:
                with open(claim_alignment_file, "r") as f:
                    details["claim_alignment"] = json.load(f)
            except Exception:
                pass

        narrative_map_file = paper_path / "quality" / "narrative_map.md"
        if narrative_map_file.exists():
            try:
                details["narrative_map"] = narrative_map_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        result_story_file = paper_path / "quality" / "result_story.md"
        if result_story_file.exists():
            try:
                details["result_story"] = result_story_file.read_text(encoding="utf-8")
            except Exception:
                pass

        contribution_map_file = paper_path / "quality" / "contribution_map_final.json"
        if contribution_map_file.exists():
            try:
                with open(contribution_map_file, "r") as f:
                    details["contribution_map"] = json.load(f)
            except Exception:
                pass

        evidence_file = paper_path / "quality" / "evidence_pack_final.json"
        if evidence_file.exists():
            try:
                with open(evidence_file, "r") as f:
                    details["evidence_pack"] = json.load(f)
            except Exception:
                pass

        key_results_file = paper_path / "quality" / "key_results_final.json"
        if key_results_file.exists():
            try:
                with open(key_results_file, "r") as f:
                    details["key_results"] = json.load(f)
            except Exception:
                pass

        submission_dashboard_file = paper_path / "quality" / "submission_dashboard.md"
        if submission_dashboard_file.exists():
            try:
                details["submission_dashboard"] = submission_dashboard_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        risk_register_file = paper_path / "quality" / "risk_register.md"
        if risk_register_file.exists():
            try:
                details["risk_register"] = risk_register_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        cover_letter_file = paper_path / "quality" / "cover_letter.md"
        if cover_letter_file.exists():
            try:
                details["cover_letter"] = cover_letter_file.read_text(encoding="utf-8")
            except Exception:
                pass

        abstract_polish_file = paper_path / "quality" / "abstract_polish.md"
        if abstract_polish_file.exists():
            try:
                details["abstract_polish"] = abstract_polish_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        rebuttal_file = paper_path / "quality" / "rebuttal_package.md"
        if rebuttal_file.exists():
            try:
                details["rebuttal_package"] = rebuttal_file.read_text(encoding="utf-8")
            except Exception:
                pass

        risk_language_plan_file = paper_path / "quality" / "risk_language_plan.md"
        if risk_language_plan_file.exists():
            try:
                details["risk_language_plan"] = risk_language_plan_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        claim_softening_plan_file = paper_path / "quality" / "claim_softening_plan.md"
        if claim_softening_plan_file.exists():
            try:
                details["claim_softening_plan"] = claim_softening_plan_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        rewrite_effectiveness_file = paper_path / "quality" / "rewrite_effectiveness.md"
        if rewrite_effectiveness_file.exists():
            try:
                details["rewrite_effectiveness"] = rewrite_effectiveness_file.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

        rewrite_trace_summary_file = (
            paper_path / "quality" / "rewrite_trace_summary.json"
        )
        if rewrite_trace_summary_file.exists():
            try:
                with open(rewrite_trace_summary_file, "r") as f:
                    details["rewrite_trace_summary"] = json.load(f)
            except Exception:
                pass

        return details

    def rebuild_index(self) -> Dict:
        """重建输出索引"""
        return rebuild_run_index(self.research_dir)

    def get_index_summary(self) -> Dict:
        """获取索引摘要"""
        index = load_run_index(self.research_dir)
        entries = index.get("entries", {})
        by_category = {}
        for entry in entries.values():
            category = entry.get("category", "unknown")
            by_category[category] = by_category.get(category, 0) + 1
        return {
            "path": str(run_index_path(self.research_dir)),
            "generated_at": index.get("generated_at"),
            "entries": len(entries),
            "by_category": by_category,
        }

    def readiness_benchmark(
        self,
        *,
        target_venue: str = "nature",
        max_entries: int = 200,
        top_n: int = 10,
        include_other_venues: bool = False,
    ) -> Dict:
        return build_readiness_benchmark(
            self.research_dir,
            target_venue=target_venue,
            max_entries=max_entries,
            top_n=top_n,
            include_other_venues=include_other_venues,
        )

    def export_readiness_benchmark(self, benchmark: Dict, output_path: str) -> str:
        return export_readiness_benchmark_markdown(benchmark, output_path)

    def shortlist_papers(
        self,
        paper_type: str = None,
        target_venue: str = None,
        require_gate: bool = False,
        require_ready: bool = False,
        min_breakthrough: float = None,
        min_priority: float = None,
        max_blockers: int = None,
        min_rewrite_gain: float = None,
        max_fallbacks: int | None = None,
        max_strict_fallbacks: int | None = 0,
        max_blocked_stages: int | None = 0,
        max_missing_stages: int | None = None,
        max_attention_stages: int | None = None,
        min_stage_score: float | None = None,
        max_self_evolution_required_failures: int | None = 0,
        min_self_evolution_score: float | None = None,
        allow_blocked_self_evolution: bool = False,
        max_blocked_processes: int | None = 0,
        min_process_alignment_score: float | None = None,
        top_n: int = 5,
    ) -> List[Dict]:
        papers = self.list_papers(paper_type=paper_type, sort_by="quality")
        filtered = []
        for paper in papers:
            if not self._passes_submission_filters(
                paper,
                target_venue=target_venue,
                require_gate=require_gate,
                require_ready=require_ready,
                min_breakthrough=min_breakthrough,
                min_priority=min_priority,
                max_blockers=max_blockers,
                min_rewrite_gain=min_rewrite_gain,
                max_fallbacks=max_fallbacks,
                max_strict_fallbacks=max_strict_fallbacks,
                max_blocked_stages=max_blocked_stages,
                max_missing_stages=max_missing_stages,
                max_attention_stages=max_attention_stages,
                min_stage_score=min_stage_score,
                max_self_evolution_required_failures=max_self_evolution_required_failures,
                min_self_evolution_score=min_self_evolution_score,
                allow_blocked_self_evolution=allow_blocked_self_evolution,
                max_blocked_processes=max_blocked_processes,
                min_process_alignment_score=min_process_alignment_score,
            ):
                continue
            filtered.append(paper)
        return filtered[:top_n]

    def submission_board(
        self,
        top_n_per_venue: int = 3,
        min_breakthrough: float = None,
        min_priority: float = None,
        max_blockers: int = None,
        min_rewrite_gain: float = None,
        require_gate: bool = False,
        max_fallbacks: int | None = None,
        max_strict_fallbacks: int | None = 0,
        max_blocked_stages: int | None = 0,
        max_missing_stages: int | None = None,
        max_attention_stages: int | None = None,
        min_stage_score: float | None = None,
        max_self_evolution_required_failures: int | None = 0,
        min_self_evolution_score: float | None = None,
        allow_blocked_self_evolution: bool = False,
        max_blocked_processes: int | None = 0,
        min_process_alignment_score: float | None = None,
    ) -> Dict[str, List[Dict]]:
        papers = self.list_papers(sort_by="quality")
        board = {}
        for paper in papers:
            if not self._passes_submission_filters(
                paper,
                min_breakthrough=min_breakthrough,
                min_priority=min_priority,
                max_blockers=max_blockers,
                min_rewrite_gain=min_rewrite_gain,
                require_gate=require_gate,
                max_fallbacks=max_fallbacks,
                max_strict_fallbacks=max_strict_fallbacks,
                max_blocked_stages=max_blocked_stages,
                max_missing_stages=max_missing_stages,
                max_attention_stages=max_attention_stages,
                min_stage_score=min_stage_score,
                max_self_evolution_required_failures=max_self_evolution_required_failures,
                min_self_evolution_score=min_self_evolution_score,
                allow_blocked_self_evolution=allow_blocked_self_evolution,
                max_blocked_processes=max_blocked_processes,
                min_process_alignment_score=min_process_alignment_score,
            ):
                continue
            venue = paper.get("target_venue") or "unknown"
            board.setdefault(venue, [])
            if len(board[venue]) < top_n_per_venue:
                board[venue].append(paper)
        return board

    def export_submission_board_markdown(
        self, board: Dict[str, List[Dict]], output_path: str
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_submission_board_markdown(board), encoding="utf-8")
        return str(output)

    def rewrite_board(
        self,
        top_n: int = 10,
        paper_type: str = None,
        target_venue: str = None,
        min_priority: float = None,
        min_rewrite_gain: float = None,
        max_blockers: int = None,
        require_gate: bool = False,
        include_ready: bool = False,
    ) -> List[Dict]:
        papers = self.list_papers(paper_type=paper_type, sort_by="quality")
        filtered = []
        for paper in papers:
            if not self._passes_submission_filters(
                paper,
                target_venue=target_venue,
                require_gate=require_gate,
                min_priority=min_priority,
                max_blockers=max_blockers,
                min_rewrite_gain=min_rewrite_gain,
            ):
                continue
            if not include_ready and paper.get("submission_status") == "ready":
                continue
            paper = dict(paper)
            paper["suggested_next_step"] = self._suggest_rewrite_next_step(paper)
            filtered.append(paper)
        filtered.sort(key=self._rewrite_board_sort_key, reverse=True)
        return filtered[:top_n]

    def export_rewrite_board_markdown(
        self, papers: List[Dict], output_path: str
    ) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_rewrite_board_markdown(papers), encoding="utf-8")
        return str(output)

    def repair_board(
        self,
        *,
        top_n: int = 20,
        target_venue: str | None = None,
        priority_tier: str | None = None,
        only_ready: bool = False,
    ) -> List[Dict]:
        rows: List[Dict] = []
        index_entries = self._get_index_entries()
        lane_fallback = {
            "figure": ("figure_repair", "Figure Repair Lane"),
            "claim": ("claim_repair", "Claim Repair Lane"),
            "section": ("section_rewrite", "Section Rewrite Lane"),
        }
        for project_root in self._iter_pipeline_projects():
            rel_path = self._relative_output_path(project_root)
            index_entry = index_entries.get(rel_path, {})
            if target_venue and index_entry.get("target_venue") != target_venue:
                continue
            review_state = (
                load_contract_artifact(
                    project_root,
                    "review_state",
                    default={},
                )
                or {}
            )
            repair_plan = (
                load_contract_artifact(
                    project_root,
                    "repair_plan",
                    default={},
                )
                or {}
            )
            repair_queue = [
                item
                for item in (repair_plan.get("tasks") or [])
                if isinstance(item, dict)
            ]
            if not repair_queue:
                repair_queue = [
                    item
                    for item in (review_state.get("repair_queue") or [])
                    if isinstance(item, dict)
                ]
            for item in repair_queue:
                if priority_tier and str(item.get("priority_tier") or "") != str(
                    priority_tier
                ):
                    continue
                if only_ready and str(item.get("status") or "") != "ready":
                    continue
                primary_target_type = str(item.get("primary_target_type") or "").strip()
                lane = str(item.get("lane") or "").strip()
                lane_label = str(item.get("lane_label") or "").strip()
                if not lane:
                    lane, lane_label = lane_fallback.get(
                        primary_target_type,
                        ("triage", "Issue Triage Lane"),
                    )
                rows.append(
                    {
                        "project": project_root.name,
                        "project_root": str(project_root),
                        "name": index_entry.get("name") or project_root.name,
                        "target_venue": index_entry.get("target_venue"),
                        "submission_priority_score": index_entry.get(
                            "submission_priority_score"
                        ),
                        "review_resolution_rate": index_entry.get(
                            "review_resolution_rate"
                        ),
                        "issue_id": item.get("issue_id"),
                        "repair_id": item.get("repair_id"),
                        "issue_text": item.get("issue_text"),
                        "role": item.get("role"),
                        "severity": item.get("severity"),
                        "status": item.get("status"),
                        "priority_tier": item.get("priority_tier"),
                        "priority_score": item.get("priority_score"),
                        "primary_target_type": item.get("primary_target_type"),
                        "primary_target_id": item.get("primary_target_id"),
                        "primary_target_label": item.get("primary_target_label"),
                        "claim_ids": list(item.get("claim_ids") or []),
                        "figure_ids": list(item.get("figure_ids") or []),
                        "section_ids": list(item.get("section_ids") or []),
                        "lane": lane,
                        "lane_label": lane_label,
                        "repair_actions": list(
                            item.get("execution_steps")
                            or item.get("repair_actions")
                            or []
                        ),
                        "verification_checks": list(
                            item.get("verification_checks") or []
                        ),
                        "blocking_reasons": list(item.get("blocking_reasons") or []),
                    }
                )
        status_rank = {
            "needs_targeting": 0,
            "needs_actions": 1,
            "needs_verification": 2,
            "ready": 3,
        }
        rows.sort(
            key=lambda item: (
                {"p0": 0, "p1": 1, "p2": 2}.get(
                    str(item.get("priority_tier") or "p2"), 3
                ),
                status_rank.get(str(item.get("status") or "ready"), 4),
                -int(item.get("priority_score") or 0),
                -float(item.get("submission_priority_score") or 0.0),
                item.get("project") or "",
                item.get("issue_id") or "",
            )
        )
        return rows[:top_n]

    def export_repair_board_markdown(self, rows: List[Dict], output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_repair_board_markdown(rows), encoding="utf-8")
        return str(output)

    def export_submission_dossier(self, paper_folder: str, output_dir: str) -> Dict:
        paper_path = self.papers_dir / paper_folder
        if not paper_path.exists():
            return {
                "status": "failed",
                "reason": f"paper folder not found: {paper_folder}",
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        copied = []
        candidates = [
            paper_path / "paper.pdf",
            paper_path / "idea.json",
            paper_path / "quality" / "submission_package.md",
            paper_path / "quality" / "logic_check_report.md",
            paper_path / "quality" / "reviewer_gate_report.md",
            paper_path / "quality" / "experiment_analysis.md",
            paper_path / "quality" / "experiment_visualization_brief.md",
            paper_path / "quality" / "figure_caption_guidance.md",
            paper_path / "quality" / "table_caption_guidance.md",
            paper_path / "quality" / "architecture_figure_brief.md",
            paper_path / "quality" / "humanizer_style_notes.md",
            paper_path / "quality" / "writing_skill_pack.md",
            paper_path / "quality" / "narrative_map.md",
            paper_path / "quality" / "result_story.md",
            paper_path / "quality" / "contribution_bullets.md",
            paper_path / "quality" / "strongest_claims.md",
            paper_path / "quality" / "editor_pitch.md",
            paper_path / "quality" / "impact_brief.md",
            paper_path / "quality" / "risk_register.md",
            paper_path / "quality" / "submission_dashboard.md",
            paper_path / "quality" / "risk_language_plan.md",
            paper_path / "quality" / "claim_softening_plan.md",
            paper_path / "quality" / "rewrite_effectiveness.md",
            paper_path / "quality" / "rewrite_trace_summary.json",
            paper_path / "quality" / "cover_letter.md",
            paper_path / "quality" / "abstract_polish.md",
            paper_path / "quality" / "rebuttal_package.md",
            paper_path / "quality" / "claim_alignment_final.json",
            paper_path / "quality" / "contribution_map_final.json",
            paper_path / "quality" / "evidence_pack_final.json",
            paper_path / "quality" / "key_results_final.json",
            paper_path / "quality" / "high_quality_result.json",
            paper_path / "experiment_todo.json",
            paper_path / "experiment_todo.md",
            paper_path / "experiment_todo_progress.json",
            paper_path / "experiment_todo_progress.md",
        ]
        for src in candidates:
            if src.exists():
                dst = output_path / src.name
                shutil.copy(src, dst)
                copied.append(str(dst))

        manifest = {
            "paper_folder": paper_folder,
            "source": str(paper_path),
            "files": copied,
        }
        manifest_path = output_path / "dossier_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        return {
            "status": "success",
            "output_dir": str(output_path),
            "manifest": str(manifest_path),
            "files": copied,
        }

    def export_shortlist_markdown(self, papers: List[Dict], output_path: str) -> str:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(render_shortlist_markdown(papers), encoding="utf-8")
        return str(output)


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def main(argv=None):
    from ai_scientist.apps.manager_cli import main as cli_main

    return cli_main(
        argv,
        manager_cls=ResearchManager,
        require_login_fn=require_login,
        resolve_output_path_fn=resolve_output_path,
        run_index_path_fn=run_index_path,
        format_size_fn=format_size,
    )


if __name__ == "__main__":
    main()
