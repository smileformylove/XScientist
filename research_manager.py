#!/usr/bin/env python3
"""
XScientist research management tool
用于管理 ./research_output 目录中的论文和批次
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目根目录到路径
PROJECT_ROOT = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

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


class ResearchManager:
    """研究管理器"""

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

    @staticmethod
    def _submission_priority_sort_key(paper: Dict):
        unresolved_critical = (
            paper.get("self_review_unresolved_critical")
            if isinstance(paper.get("self_review_unresolved_critical"), int)
            else 999
        )
        blocked_stage_count = (
            int(paper.get("blocked_stage_count"))
            if isinstance(paper.get("blocked_stage_count"), int)
            else 999
        )
        missing_stage_count = (
            int(paper.get("missing_stage_count"))
            if isinstance(paper.get("missing_stage_count"), int)
            else 999
        )
        attention_stage_count = (
            int(paper.get("needs_attention_stage_count"))
            if isinstance(paper.get("needs_attention_stage_count"), int)
            else 999
        )
        stage_overall_score = (
            float(paper.get("stage_overall_score"))
            if isinstance(paper.get("stage_overall_score"), (int, float))
            else -1.0
        )
        fallback_count = (
            int(paper.get("fallback_count"))
            if isinstance(paper.get("fallback_count"), int)
            else 999
        )
        strict_fallback_count = (
            int(paper.get("strict_fallback_count"))
            if isinstance(paper.get("strict_fallback_count"), int)
            else 999
        )
        gate_score = (
            paper.get("self_review_round_gate_score")
            if isinstance(paper.get("self_review_round_gate_score"), (int, float))
            else -1
        )
        review_resolution_rate = (
            float(paper.get("review_resolution_rate"))
            if isinstance(paper.get("review_resolution_rate"), (int, float))
            else -1.0
        )
        review_active_issue_count = (
            int(paper.get("review_active_issue_count"))
            if isinstance(paper.get("review_active_issue_count"), int)
            else 999
        )
        review_persistent_issue_count = (
            int(paper.get("review_persistent_issue_count"))
            if isinstance(paper.get("review_persistent_issue_count"), int)
            else 999
        )
        review_unbound_issue_count = (
            int(paper.get("review_unbound_issue_count"))
            if isinstance(paper.get("review_unbound_issue_count"), int)
            else 999
        )
        review_target_binding_coverage = (
            float(paper.get("review_target_binding_coverage"))
            if isinstance(paper.get("review_target_binding_coverage"), (int, float))
            else -1.0
        )
        review_repair_ready_coverage = (
            float(paper.get("review_repair_ready_coverage"))
            if isinstance(paper.get("review_repair_ready_coverage"), (int, float))
            else -1.0
        )
        self_evolution_status = str(paper.get("self_evolution_status") or "").strip()
        self_evolution_score = (
            float(paper.get("self_evolution_score"))
            if isinstance(paper.get("self_evolution_score"), (int, float))
            else -1.0
        )
        self_evolution_required_failure_count = (
            int(paper.get("self_evolution_required_failure_count"))
            if isinstance(paper.get("self_evolution_required_failure_count"), int)
            else 999
        )
        process_alignment_score = (
            float(paper.get("process_alignment_overall_score"))
            if isinstance(paper.get("process_alignment_overall_score"), (int, float))
            else -1.0
        )
        process_alignment_blocked_count = (
            int(paper.get("process_alignment_blocked_process_count"))
            if isinstance(paper.get("process_alignment_blocked_process_count"), int)
            else 999
        )
        return (
            paper.get("submission_status") == "ready",
            paper.get("quality_gate_passed") is True,
            paper.get("self_review_round_gate_ready") is True,
            blocked_stage_count == 0,
            missing_stage_count == 0,
            attention_stage_count == 0,
            process_alignment_blocked_count == 0,
            self_evolution_status != "blocked",
            gate_score,
            stage_overall_score,
            process_alignment_score,
            self_evolution_score,
            review_resolution_rate,
            review_repair_ready_coverage,
            review_target_binding_coverage,
            -process_alignment_blocked_count,
            -self_evolution_required_failure_count,
            -review_active_issue_count,
            -review_persistent_issue_count,
            -review_unbound_issue_count,
            -blocked_stage_count,
            -missing_stage_count,
            -attention_stage_count,
            -strict_fallback_count,
            -fallback_count,
            -unresolved_critical,
            (
                paper.get("submission_priority_score")
                if isinstance(paper.get("submission_priority_score"), (int, float))
                else -1
            ),
            (
                paper.get("rewrite_priority_gain_total")
                if isinstance(paper.get("rewrite_priority_gain_total"), (int, float))
                else -999
            ),
            -(
                paper.get("blocker_count")
                if isinstance(paper.get("blocker_count"), int)
                else 999
            ),
            -(
                paper.get("critical_revision_actions_count")
                if isinstance(paper.get("critical_revision_actions_count"), int)
                else 999
            ),
            (
                paper.get("quality_score")
                if isinstance(paper.get("quality_score"), (int, float))
                else -1
            ),
            (
                paper.get("breakthrough_score")
                if isinstance(paper.get("breakthrough_score"), (int, float))
                else -1
            ),
            (
                paper.get("rigor_score")
                if isinstance(paper.get("rigor_score"), (int, float))
                else -1
            ),
            (
                paper.get("claim_support_score")
                if isinstance(paper.get("claim_support_score"), (int, float))
                else -1
            ),
            (
                paper.get("numeric_coverage_score")
                if isinstance(paper.get("numeric_coverage_score"), (int, float))
                else -1
            ),
            (
                paper.get("evidence_density_score")
                if isinstance(paper.get("evidence_density_score"), (int, float))
                else -1
            ),
            (
                paper.get("contribution_count")
                if isinstance(paper.get("contribution_count"), int)
                else -1
            ),
            -(
                paper.get("unsupported_claims_count")
                if isinstance(paper.get("unsupported_claims_count"), int)
                else 999
            ),
            paper.get("modified_at", ""),
        )

    @staticmethod
    def _passes_submission_filters(
        paper: Dict,
        *,
        target_venue: str = None,
        require_gate: bool = False,
        require_ready: bool = False,
        min_breakthrough: float = None,
        min_priority: float = None,
        max_blockers: int = None,
        min_rewrite_gain: float = None,
        max_fallbacks: int = None,
        max_strict_fallbacks: int = None,
        max_blocked_stages: int = None,
        max_missing_stages: int = None,
        max_attention_stages: int = None,
        min_stage_score: float = None,
        max_self_evolution_required_failures: int | None = 0,
        min_self_evolution_score: float | None = None,
        allow_blocked_self_evolution: bool = False,
        max_blocked_processes: int | None = 0,
        min_process_alignment_score: float | None = None,
    ) -> bool:
        if target_venue and paper.get("target_venue") != target_venue:
            return False
        if require_gate and paper.get("quality_gate_passed") is not True:
            return False
        if require_ready and paper.get("submission_status") != "ready":
            return False
        if (
            min_breakthrough is not None
            and (paper.get("breakthrough_score") or 0) < min_breakthrough
        ):
            return False
        if (
            min_priority is not None
            and (paper.get("submission_priority_score") or 0) < min_priority
        ):
            return False
        if (
            max_blockers is not None
            and isinstance(paper.get("blocker_count"), int)
            and paper.get("blocker_count") > max_blockers
        ):
            return False
        if (
            min_rewrite_gain is not None
            and (paper.get("rewrite_priority_gain_total") or 0) < min_rewrite_gain
        ):
            return False
        if (
            max_fallbacks is not None
            and isinstance(paper.get("fallback_count"), int)
            and paper.get("fallback_count") > max_fallbacks
        ):
            return False
        if (
            max_strict_fallbacks is not None
            and isinstance(paper.get("strict_fallback_count"), int)
            and paper.get("strict_fallback_count") > max_strict_fallbacks
        ):
            return False
        if (
            max_blocked_stages is not None
            and isinstance(paper.get("blocked_stage_count"), int)
            and paper.get("blocked_stage_count") > max_blocked_stages
        ):
            return False
        if (
            max_missing_stages is not None
            and isinstance(paper.get("missing_stage_count"), int)
            and paper.get("missing_stage_count") > max_missing_stages
        ):
            return False
        if (
            max_attention_stages is not None
            and isinstance(paper.get("needs_attention_stage_count"), int)
            and paper.get("needs_attention_stage_count") > max_attention_stages
        ):
            return False
        if (
            min_stage_score is not None
            and isinstance(paper.get("stage_overall_score"), (int, float))
            and float(paper.get("stage_overall_score")) < min_stage_score
        ):
            return False
        if (
            not allow_blocked_self_evolution
            and str(paper.get("self_evolution_status") or "").strip() == "blocked"
        ):
            return False
        if (
            max_self_evolution_required_failures is not None
            and isinstance(paper.get("self_evolution_required_failure_count"), int)
            and paper.get("self_evolution_required_failure_count")
            > max_self_evolution_required_failures
        ):
            return False
        if (
            min_self_evolution_score is not None
            and isinstance(paper.get("self_evolution_score"), (int, float))
            and float(paper.get("self_evolution_score")) < min_self_evolution_score
        ):
            return False
        if (
            max_blocked_processes is not None
            and isinstance(paper.get("process_alignment_blocked_process_count"), int)
            and paper.get("process_alignment_blocked_process_count")
            > max_blocked_processes
        ):
            return False
        if (
            min_process_alignment_score is not None
            and isinstance(paper.get("process_alignment_overall_score"), (int, float))
            and float(paper.get("process_alignment_overall_score"))
            < min_process_alignment_score
        ):
            return False
        return True

    @staticmethod
    def _rewrite_board_sort_key(paper: Dict):
        gate_ready = paper.get("self_review_round_gate_ready")
        gate_score = (
            float(paper.get("self_review_round_gate_score"))
            if isinstance(paper.get("self_review_round_gate_score"), (int, float))
            else (100.0 if gate_ready is True else 0.0 if gate_ready is False else 50.0)
        )
        gate_deficit = max(0.0, 100.0 - gate_score)
        unresolved_critical = (
            int(paper.get("self_review_unresolved_critical"))
            if isinstance(paper.get("self_review_unresolved_critical"), int)
            else 0
        )
        high_value_coverage = (
            float(paper.get("self_review_high_value_coverage"))
            if isinstance(paper.get("self_review_high_value_coverage"), (int, float))
            else 1.0
        )
        high_value_gap = max(0.0, 1.0 - high_value_coverage)
        focus_issue_count = (
            int(paper.get("self_review_focus_issue_count"))
            if isinstance(paper.get("self_review_focus_issue_count"), int)
            else 0
        )
        review_active_issue_count = (
            int(paper.get("review_active_issue_count"))
            if isinstance(paper.get("review_active_issue_count"), int)
            else 0
        )
        review_persistent_issue_count = (
            int(paper.get("review_persistent_issue_count"))
            if isinstance(paper.get("review_persistent_issue_count"), int)
            else 0
        )
        review_unbound_issue_count = (
            int(paper.get("review_unbound_issue_count"))
            if isinstance(paper.get("review_unbound_issue_count"), int)
            else 0
        )
        review_active_binding_coverage = (
            float(paper.get("review_active_binding_coverage"))
            if isinstance(paper.get("review_active_binding_coverage"), (int, float))
            else 1.0
        )
        review_repair_ready_coverage = (
            float(paper.get("review_repair_ready_coverage"))
            if isinstance(paper.get("review_repair_ready_coverage"), (int, float))
            else 0.0
        )
        review_resolution_rate = (
            float(paper.get("review_resolution_rate"))
            if isinstance(paper.get("review_resolution_rate"), (int, float))
            else 0.0
        )
        experiment_todo_count = (
            int(paper.get("experiment_todo_count"))
            if isinstance(paper.get("experiment_todo_count"), int)
            else 0
        )
        experiment_todo_p0_count = (
            int(paper.get("experiment_todo_p0_count"))
            if isinstance(paper.get("experiment_todo_p0_count"), int)
            else 0
        )
        experiment_todo_closure_rate = (
            float(paper.get("experiment_todo_closure_rate"))
            if isinstance(paper.get("experiment_todo_closure_rate"), (int, float))
            else 1.0
        )
        experiment_todo_closure_gap = max(0.0, 1.0 - experiment_todo_closure_rate)
        return (
            paper.get("submission_status") != "ready",
            gate_ready is False,
            gate_deficit,
            unresolved_critical,
            high_value_gap,
            review_persistent_issue_count,
            review_active_issue_count,
            review_unbound_issue_count,
            max(0.0, 1.0 - review_repair_ready_coverage),
            max(0.0, 1.0 - review_active_binding_coverage),
            max(0.0, 1.0 - review_resolution_rate),
            focus_issue_count,
            experiment_todo_p0_count,
            experiment_todo_count,
            experiment_todo_closure_gap,
            (
                paper.get("rewrite_priority_gain_total")
                if isinstance(paper.get("rewrite_priority_gain_total"), (int, float))
                else -999
            ),
            (
                paper.get("rewrite_best_round_priority_delta")
                if isinstance(
                    paper.get("rewrite_best_round_priority_delta"), (int, float)
                )
                else -999
            ),
            (
                paper.get("submission_priority_score")
                if isinstance(paper.get("submission_priority_score"), (int, float))
                else -1
            ),
            -(
                paper.get("blocker_count")
                if isinstance(paper.get("blocker_count"), int)
                else 999
            ),
            (
                paper.get("quality_score")
                if isinstance(paper.get("quality_score"), (int, float))
                else -1
            ),
            paper.get("modified_at", ""),
        )

    @staticmethod
    def _suggest_rewrite_next_step(paper: Dict) -> str:
        gate_ready = paper.get("self_review_round_gate_ready")
        gate_reasons = paper.get("self_review_round_gate_reasons") or []
        next_focus = paper.get("self_review_next_focus") or []
        unresolved_critical = paper.get("self_review_unresolved_critical")
        experiment_todo_count = (
            int(paper.get("experiment_todo_count"))
            if isinstance(paper.get("experiment_todo_count"), int)
            else 0
        )
        experiment_todo_p0_count = (
            int(paper.get("experiment_todo_p0_count"))
            if isinstance(paper.get("experiment_todo_p0_count"), int)
            else 0
        )
        experiment_todo_top_action = str(
            paper.get("experiment_todo_top_action") or ""
        ).strip()
        review_persistent_issue_count = (
            int(paper.get("review_persistent_issue_count"))
            if isinstance(paper.get("review_persistent_issue_count"), int)
            else 0
        )
        review_active_issue_count = (
            int(paper.get("review_active_issue_count"))
            if isinstance(paper.get("review_active_issue_count"), int)
            else 0
        )
        review_resolution_rate = (
            float(paper.get("review_resolution_rate"))
            if isinstance(paper.get("review_resolution_rate"), (int, float))
            else None
        )
        review_repair_ready_coverage = (
            float(paper.get("review_repair_ready_coverage"))
            if isinstance(paper.get("review_repair_ready_coverage"), (int, float))
            else None
        )
        experiment_todo_closure_rate = (
            float(paper.get("experiment_todo_closure_rate"))
            if isinstance(paper.get("experiment_todo_closure_rate"), (int, float))
            else None
        )
        if gate_ready is False:
            if isinstance(unresolved_critical, int) and unresolved_critical > 0:
                return (
                    "Round gate still reports unresolved critical issues; fix critical soundness/evidence gaps before polish."
                )
            if "high_value_coverage_low" in gate_reasons:
                if next_focus:
                    return (
                        f"Prioritize unresolved high-value issues first: {next_focus[0]}"
                    )
                return (
                    "High-value issue coverage is still low; target P0/P1 issues before broad rewrites."
                )
            if "rewrite_coverage_low" in gate_reasons:
                return (
                    "Increase issue-linked rewrite coverage and ensure addressed_issue_ids match recommended targets."
                )
            if "persistent_issues_high" in gate_reasons and next_focus:
                return (
                    f"Persistent issues remain; continue with focused repair on: {next_focus[0]}"
                )
            if next_focus:
                return f"Continue round-gate focus: {next_focus[0]}"
        if review_persistent_issue_count > 0:
            return (
                "Reviewer issues are persisting across rounds; do focused issue-by-issue repairs with explicit verification before broader rewrites."
            )
        if (
            review_active_issue_count > 0
            and review_repair_ready_coverage is not None
            and review_repair_ready_coverage < 1.0
        ):
            return (
                "Some active reviewer issues still lack fully executable repair tasks; expand the repair queue with concrete actions and target-specific verification."
            )
        if review_active_issue_count > 0 and review_unbound_issue_count > 0:
            return (
                "Some reviewer issues are not yet mapped to a claim, figure, or section; bind them first so the next repair round is targeted."
            )
        if review_active_issue_count > 0 and review_resolution_rate is not None and review_resolution_rate < 0.35:
            return (
                "Reviewer debt is still high relative to resolved issues; prioritize concrete fixes and evidence checks over stylistic polish."
            )
        if experiment_todo_p0_count > 0 and experiment_todo_top_action:
            return (
                f"Execute the highest-priority experiment TODO first: "
                f"{experiment_todo_top_action}"
            )
        if experiment_todo_p0_count > 0:
            return "Resolve P0 experiment TODO items before broad stylistic rewrites."
        if experiment_todo_count > 0 and experiment_todo_top_action:
            return (
                f"Start with open experiment TODO item: "
                f"{experiment_todo_top_action}"
            )
        if (
            experiment_todo_count > 0
            and isinstance(experiment_todo_closure_rate, float)
            and experiment_todo_closure_rate < 0.5
        ):
            return (
                "Experiment TODO closure rate remains low; prioritize one measurable evidence task this round."
            )
        if paper.get("submission_status") == "ready":
            return "Ready or near-ready; do final polish and package review."
        if paper.get("rewrite_top_section") and paper.get(
            "rewrite_top_frontmatter_style"
        ):
            return (
                f"Continue with {paper.get('rewrite_top_section')} using the "
                f"'{paper.get('rewrite_top_frontmatter_style')}' frontmatter framing style."
            )
        if paper.get("rewrite_top_section"):
            return f"Continue targeted rewriting on {paper.get('rewrite_top_section')}."
        if (
            isinstance(paper.get("blocker_count"), int)
            and paper.get("blocker_count") > 4
        ):
            return "Too many blockers remain; reduce blockers before spending more rewrite budget."
        if (
            isinstance(paper.get("rewrite_priority_gain_total"), (int, float))
            and paper.get("rewrite_priority_gain_total") > 1.0
        ):
            return "One more rewrite pass looks worthwhile; recent rewrites are still improving submission priority."
        if gate_ready is False:
            return "Round gate not yet ready; prioritize unresolved high-value self-review issues."
        return "Review the risk-language and claim-softening plans before the next rewrite pass."

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
            research_plan = load_contract_artifact(
                project_root,
                "research_plan",
                default={},
            ) or {}
            review_state = load_contract_artifact(
                project_root,
                "review_state",
                default={},
            ) or {}
            repair_plan = load_contract_artifact(
                project_root,
                "repair_plan",
                default={},
            ) or {}
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
                research_plan.get("budget")
                if isinstance(research_plan, dict)
                else {}
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
                    "acceptance_rule_count": len(execution_policy.get("acceptance_rules") or []),
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
                        review_repair_metrics.get(
                            "repair_verification_ready_coverage"
                        )
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
                    "self_evolution_file": str(project_root / "self_evolution.json")
                    if (project_root / "self_evolution.json").exists()
                    else None,
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
                    "process_alignment_file": str(
                        project_root / "process_alignment.json"
                    )
                    if (project_root / "process_alignment.json").exists()
                    else None,
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
                        (process_alignment_summary.get("top_process_risks") or {}).keys()
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
                    "warnings": sorted(set(str(item) for item in warnings if str(item).strip())),
                    "modified_at": datetime.fromtimestamp(project_root.stat().st_mtime).isoformat(),
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
                item for item in (evolution.get("lessons") or []) if isinstance(item, dict)
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
                        "required_failures": list(result.get("required_failures") or []),
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
                        "required_failures": list(result.get("required_failures") or []),
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
            idea_cards = load_contract_artifact(project_root, "idea_cards", default=[]) or []
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
                        "minimum_viable_experiment": card.get("minimum_viable_experiment"),
                        "modified_at": datetime.fromtimestamp(project_root.stat().st_mtime).isoformat(),
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
                or str(item.get("self_evolution_required_failure_count") or "").isdigit()
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
            fallback_run_count = sum(count > 0 for count in fallback_counts) if fallback_counts else 0
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
                    "self_evolution_next_cycle_stage_counts": dict(next_cycle_stage_counts),
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
                    "latest_project": latest.get("project") or latest.get("relative_path"),
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
                    "source": heaviest.get("source_name")
                    or heaviest.get("source_key"),
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

        dominant_archetype, dominant_archetype_count = archetype_counts.most_common(1)[0]
        if len(archetype_counts) <= 2 and dominant_archetype_count >= max(3, len(rows) - 1):
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
                "top_self_evolution_risks": dict(
                    evolution_risk_counts.most_common(6)
                ),
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
            diversify_reason = (
                "Keep a second research posture warm so the next batch is not overfit to one source."
            )
            if "mix_too_narrow" in recommendation_labels and dominant_archetype:
                diversify_reason = (
                    f"The current portfolio is dominated by {dominant_archetype}, so this lane widens the batch mix."
                )
            elif desired_policy and (
                diversify.get("source_workflow_mode") == desired_policy
                or diversify.get("source_archetype")
                == desired_to_archetype.get(str(desired_policy))
            ):
                diversify_reason = (
                    f"This lane improves coverage for the active {desired_policy} policy."
                )
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
            hardening_reason = (
                "Reserve one lane for evidence packaging, review hardening, or submission-grade convergence."
            )
            if hardening.get("source_archetype") == "review_hardening":
                hardening_reason = (
                    "This lane raises reviewer-facing pressure before the next submission push."
                )
            elif hardening.get("source_archetype") == "writing_polish":
                hardening_reason = (
                    "This lane focuses the batch on figures, captions, and writing polish."
                )
            elif hardening.get("source_archetype") == "program_guarded":
                hardening_reason = (
                    "This lane keeps the batch tied to a budgeted, submission-oriented program."
                )
            if int(hardening.get("blocked_self_evolution_run_count") or 0) > 0:
                hardening_reason += (
                    " It also helps absorb self-evolution debt before more open exploration."
                )
            add_slot(
                "hardening_lane",
                0.2,
                hardening,
                hardening_reason,
                "Ensure at least one lane converges toward submission-grade artifacts.",
            )

        cadence_label = "balanced_rotation"
        cadence_reason = (
            "Keep a balanced portfolio so the next batch can discover, validate, and polish in parallel."
        )
        if "missing_desired_policy_source" in recommendation_labels:
            cadence_label = "portfolio_rebalance"
            cadence_reason = (
                f"The active {desired_policy} policy is underrepresented, so the next batch should rebalance the source portfolio before scaling throughput."
            )
        elif "mix_too_narrow" in recommendation_labels:
            cadence_label = "rebalance_then_converge"
            cadence_reason = (
                f"The mix is currently concentrated in {dominant_archetype}; use the next batch to widen the portfolio before converging."
            )
        elif desired_policy == "agentic_tree":
            cadence_label = "explore_then_converge"
            cadence_reason = (
                "Favor a wider first lane, then converge with one harder evidence or review lane."
            )
        elif desired_policy in {
            "program_driven",
            "review_board",
            "writing_studio",
            "multi_agent_board",
        }:
            cadence_label = "submission_hardening_loop"
            cadence_reason = (
                f"Treat the next batch as a {desired_policy} convergence pass with tighter evidence and review discipline."
            )
        elif desired_policy == "classic_pipeline":
            cadence_label = "repeatable_throughput"
            cadence_reason = (
                "Favor stable, repeatable sources while keeping one smaller secondary lane alive."
            )

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
        lines = ["# Submission Board", ""]
        for venue, papers in sorted(board.items()):
            lines.append(f"## {venue}")
            for paper in papers:
                lines.append(
                    f"- {paper['name']} | priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                    f"rewrite_gain={paper.get('rewrite_priority_gain_total')} | blockers={paper.get('blocker_count')} | "
                    f"stage_score={paper.get('stage_overall_score')} blocked_stages={paper.get('blocked_stage_count')} "
                    f"attention_stages={paper.get('needs_attention_stage_count')} missing_stages={paper.get('missing_stage_count')} | "
                    f"self_evolution={paper.get('self_evolution_status')} score={paper.get('self_evolution_score')} "
                    f"required_failures={paper.get('self_evolution_required_failure_count')} | "
                    f"process_alignment={paper.get('process_alignment_overall_score')} blocked_processes={paper.get('process_alignment_blocked_process_count')} | "
                    f"review_resolution={paper.get('review_resolution_rate')} review_binding={paper.get('review_target_binding_coverage')} "
                    f"active_review_issues={paper.get('review_active_issue_count')} persistent_review_issues={paper.get('review_persistent_issue_count')} | "
                    f"fallbacks={paper.get('fallback_count')} strict={paper.get('strict_fallback_count')} | quality={paper.get('quality_score')} | "
                    f"rigor={paper.get('rigor_score')} | claim={paper.get('claim_support_score')} | "
                    f"package={paper.get('submission_package_file')}"
                )
            lines.append("")
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        lines = ["# Rewrite Board", ""]
        for paper in papers:
            lines.extend(
                [
                    f"## {paper.get('name')}",
                    f"- Venue: {paper.get('target_venue')}",
                    f"- Submission Priority: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})",
                    f"- Rewrite Gain: {paper.get('rewrite_priority_gain_total')}",
                    f"- Best Round Delta: {paper.get('rewrite_best_round_priority_delta')}",
                    f"- Rewrite Rounds: {paper.get('rewrite_round_count')}",
                    f"- Self Review Rounds: {paper.get('self_review_rounds_completed')}",
                    f"- Round Gate: ready={paper.get('self_review_round_gate_ready')} score={paper.get('self_review_round_gate_score')} unresolved_critical={paper.get('self_review_unresolved_critical')}",
                    f"- Reviewer Repair: resolution={paper.get('review_resolution_rate')} active={paper.get('review_active_issue_count')} persistent={paper.get('review_persistent_issue_count')} checks={paper.get('review_verification_count')}",
                    f"- Reviewer Repair Queue: queue={paper.get('review_repair_queue_count')} ready={paper.get('review_repair_ready_count')} ready_coverage={paper.get('review_repair_ready_coverage')} verification_ready={paper.get('review_repair_verification_ready_count')}",
                    f"- Reviewer Target Binding: coverage={paper.get('review_target_binding_coverage')} active_coverage={paper.get('review_active_binding_coverage')} unbound={paper.get('review_unbound_issue_count')}",
                    f"- Experiment TODO: total={paper.get('experiment_todo_count')} p0={paper.get('experiment_todo_p0_count')}",
                    f"- Experiment TODO Progress: closed={paper.get('experiment_todo_closed_count')} unresolved={paper.get('experiment_todo_unresolved_count')} closure_rate={paper.get('experiment_todo_closure_rate')} p0_closure_rate={paper.get('experiment_todo_p0_closure_rate')}",
                    f"- Experiment TODO Top Action: {paper.get('experiment_todo_top_action')}",
                    f"- High-Value Coverage: {paper.get('self_review_high_value_coverage')}",
                    f"- Top Section: {paper.get('rewrite_top_section')}",
                    f"- Top Section Style: {paper.get('rewrite_top_section_style')}",
                    f"- Top Frontmatter Style: {paper.get('rewrite_top_frontmatter_style')}",
                    f"- Blockers: {paper.get('blocker_count')}",
                    f"- Next Step: {paper.get('suggested_next_step')}",
                    f"- Experiment TODO File: {paper.get('experiment_todo_file')}",
                    f"- Experiment TODO Progress File: {paper.get('experiment_todo_progress_file')}",
                    f"- Rewrite Effectiveness: {paper.get('rewrite_effectiveness_file')}",
                    f"- Path: {paper.get('path')}",
                    "",
                ]
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
            review_state = load_contract_artifact(
                project_root,
                "review_state",
                default={},
            ) or {}
            repair_plan = load_contract_artifact(
                project_root,
                "repair_plan",
                default={},
            ) or {}
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
        lines = ["# Repair Board", ""]
        for row in rows:
            lines.extend(
                [
                    f"## {row.get('name')} :: {row.get('repair_id')}",
                    f"- Venue: {row.get('target_venue')}",
                    f"- Project: {row.get('project')}",
                    f"- Reviewer role: {row.get('role')}",
                    f"- Priority: {row.get('priority_tier')} ({row.get('priority_score')})",
                    f"- Status: {row.get('status')}",
                    f"- Issue: {row.get('issue_text')}",
                    f"- Target: {row.get('primary_target_type')} {row.get('primary_target_id')} ({row.get('primary_target_label')})",
                    f"- Lane: {row.get('lane')} ({row.get('lane_label')})",
                    f"- Blocking reasons: {', '.join(row.get('blocking_reasons') or []) or 'none'}",
                    f"- Repair actions: {' | '.join(row.get('repair_actions') or []) or 'none'}",
                    f"- Verification checks: {' | '.join(row.get('verification_checks') or []) or 'none'}",
                    f"- Path: {row.get('project_root')}",
                    "",
                ]
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
        lines = ["# Submission Shortlist", ""]
        for paper in papers:
            lines.extend(
                [
                    f"## {paper.get('name')}",
                    f"- Type: {paper.get('type')}",
                    f"- Venue: {paper.get('target_venue')}",
                    f"- Submission Priority: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})",
                    f"- Rewrite Priority Gain: {paper.get('rewrite_priority_gain_total')}",
                    f"- Blockers: {paper.get('blocker_count')}",
                    f"- Stage Standards: score={paper.get('stage_overall_score')} blocked={paper.get('blocked_stage_count')} attention={paper.get('needs_attention_stage_count')} missing={paper.get('missing_stage_count')}",
                    f"- Top Standard Risks: {', '.join(paper.get('top_standard_risks') or [])}",
                    f"- Process Alignment: score={paper.get('process_alignment_overall_score')} blocked={paper.get('process_alignment_blocked_process_count')} attention={paper.get('process_alignment_attention_process_count')} missing={paper.get('process_alignment_missing_process_count')}",
                    f"- Process Risks: {', '.join(paper.get('process_alignment_top_risks') or [])}",
                    f"- Reviewer Repair: resolution={paper.get('review_resolution_rate')} active={paper.get('review_active_issue_count')} persistent={paper.get('review_persistent_issue_count')} checks={paper.get('review_verification_count')}",
                    f"- Reviewer Repair Queue: queue={paper.get('review_repair_queue_count')} ready={paper.get('review_repair_ready_count')} ready_coverage={paper.get('review_repair_ready_coverage')} verification_ready={paper.get('review_repair_verification_ready_count')}",
                    f"- Reviewer Target Binding: coverage={paper.get('review_target_binding_coverage')} active_coverage={paper.get('review_active_binding_coverage')} unbound={paper.get('review_unbound_issue_count')}",
                    f"- Self-Evolution: status={paper.get('self_evolution_status')} score={paper.get('self_evolution_score')} required_failures={paper.get('self_evolution_required_failure_count')} lane={paper.get('self_evolution_dominant_lane')} role={paper.get('self_evolution_dominant_role')}",
                    f"- Self-Evolution Risks: {', '.join(paper.get('self_evolution_top_risks') or [])}",
                    f"- Quality: {paper.get('quality_score')}",
                    f"- Breakthrough: {paper.get('breakthrough_score')}",
                    f"- Rigor: {paper.get('rigor_score')}",
                    f"- Claim Support: {paper.get('claim_support_score')}",
                    f"- Numeric Coverage: {paper.get('numeric_coverage_score')}",
                    f"- Contributions: {paper.get('contribution_count')}",
                    f"- Gate Passed: {paper.get('quality_gate_passed')}",
                    f"- Submission Status: {paper.get('submission_status')}",
                    f"- Submission Package: {paper.get('submission_package_file')}",
                    f"- Narrative Map: {paper.get('narrative_map_file')}",
                    f"- Contribution Map: {paper.get('contribution_map_file')}",
                    f"- Editor Pitch: {paper.get('editor_pitch_file')}",
                    f"- Submission Dashboard: {paper.get('submission_dashboard_file')}",
                    f"- Risk Language Plan: {paper.get('risk_language_plan_file')}",
                    f"- Claim Softening Plan: {paper.get('claim_softening_plan_file')}",
                    f"- Rewrite Effectiveness: {paper.get('rewrite_effectiveness_file')}",
                    f"- Rewrite Best Round Delta: {paper.get('rewrite_best_round_priority_delta')}",
                    f"- Rewrite Top Section: {paper.get('rewrite_top_section')}",
                    f"- Risk Register: {paper.get('risk_register_file')}",
                    f"- Path: {paper.get('path')}",
                    "",
                ]
            )
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(output)


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def main():
    require_login("研究管理操作(research_manager)")

    parser = argparse.ArgumentParser(
        description="XScientist research management tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

1. 列出所有批次:
   python research_manager.py list-batches

2. 查看批次详情:
   python research_manager.py batch-summary 20240101_120000

3. 列出所有论文:
   python research_manager.py list-papers

4. 列出特定类型论文:
   python research_manager.py list-papers --type normal

5. 搜索论文:
   python research_manager.py search-papers "transformer"

6. 清理旧文件 (30天前):
   python research_manager.py cleanup --days 30

7. 预览将要清理的文件:
   python research_manager.py cleanup --days 30 --dry-run

8. 重建结果索引:
   python research_manager.py rebuild-index

9. 按质量生成投稿 shortlist:
   python research_manager.py shortlist --top 5 --require-gate

10. 查看 Nature-style readiness benchmark:
   python research_manager.py readiness-benchmark --venue nature --top 10

11. 查看 source lineage 看板:
   python research_manager.py source-board --top 20

12. 查看 source mix 建议:
   python research_manager.py source-mix --desired-policy program_driven

13. 查看 reviewer repair queue:
   python research_manager.py repair-board --top 20

14. 查看 self-evolution 看板:
   python research_manager.py evolution-board --top 20
        """,
    )

    parser.add_argument(
        "--research-dir",
        type=str,
        default=str(resolve_output_path()),
        help="研究目录路径",
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # list-batches 命令
    subparsers.add_parser("list-batches", help="列出所有批次")

    # batch-summary 命令
    batch_summary_parser = subparsers.add_parser("batch-summary", help="查看批次摘要")
    batch_summary_parser.add_argument("batch_name", help="批次名称 (不含batch_前缀)")

    # list-papers 命令
    list_papers_parser = subparsers.add_parser("list-papers", help="列出所有论文")
    list_papers_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )
    list_papers_parser.add_argument(
        "--detailed", action="store_true", help="显示详细信息"
    )
    list_papers_parser.add_argument(
        "--sort", choices=["modified", "quality"], default="modified", help="排序方式"
    )

    # paper-details 命令
    details_parser = subparsers.add_parser("paper-details", help="查看论文详细信息")
    details_parser.add_argument("folder", help="论文文件夹名称")

    # list-ideas 命令
    subparsers.add_parser("list-ideas", help="列出所有想法")

    # search-papers 命令
    search_parser = subparsers.add_parser("search-papers", help="搜索论文")
    search_parser.add_argument("query", help="搜索关键词")
    search_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )

    # cleanup 命令
    cleanup_parser = subparsers.add_parser("cleanup", help="清理旧文件")
    cleanup_parser.add_argument("--days", type=int, default=30, help="天数阈值")
    cleanup_parser.add_argument("--dry-run", action="store_true", help="仅显示，不删除")

    # stats 命令
    subparsers.add_parser("stats", help="显示统计信息")

    # rebuild-index 命令
    subparsers.add_parser("rebuild-index", help="重建输出索引")

    # shortlist 命令
    shortlist_parser = subparsers.add_parser(
        "shortlist", help="按质量筛选最值得投稿的论文"
    )
    shortlist_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )
    shortlist_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        help="按目标 venue 过滤",
    )
    shortlist_parser.add_argument(
        "--require-gate", action="store_true", help="只保留通过质量门槛的稿件"
    )
    shortlist_parser.add_argument(
        "--require-ready",
        action="store_true",
        help="只保留 submission readiness 为 ready 的稿件",
    )
    shortlist_parser.add_argument(
        "--min-breakthrough", type=float, default=None, help="最小 breakthrough score"
    )
    shortlist_parser.add_argument(
        "--min-priority",
        type=float,
        default=None,
        help="最小 submission priority score",
    )
    shortlist_parser.add_argument(
        "--max-blockers", type=int, default=None, help="最多允许的 blocker 数"
    )
    shortlist_parser.add_argument(
        "--min-rewrite-gain",
        type=float,
        default=None,
        help="最小 rewrite priority gain",
    )
    shortlist_parser.add_argument(
        "--max-fallbacks",
        type=int,
        default=None,
        help="最多允许的 fallback 事件数",
    )
    shortlist_parser.add_argument(
        "--max-strict-fallbacks",
        type=int,
        default=0,
        help="最多允许的 strict fallback 事件数，默认不接受 strict fallback",
    )
    shortlist_parser.add_argument(
        "--max-blocked-stages",
        type=int,
        default=0,
        help="最多允许的 blocked stage standards 数，默认不接受 blocked stage",
    )
    shortlist_parser.add_argument(
        "--max-missing-stages",
        type=int,
        default=None,
        help="最多允许的 missing stage standards 数",
    )
    shortlist_parser.add_argument(
        "--max-attention-stages",
        type=int,
        default=None,
        help="最多允许的 needs_attention stage standards 数",
    )
    shortlist_parser.add_argument(
        "--min-stage-score",
        type=float,
        default=None,
        help="最小 stage standards overall score",
    )
    shortlist_parser.add_argument(
        "--max-self-evolution-required-failures",
        type=int,
        default=0,
        help="最多允许的 self-evolution required failure 数，默认不接受 required failure",
    )
    shortlist_parser.add_argument(
        "--min-self-evolution-score",
        type=float,
        default=None,
        help="最小 self-evolution score",
    )
    shortlist_parser.add_argument(
        "--allow-blocked-self-evolution",
        action="store_true",
        help="允许 self-evolution status 为 blocked 的稿件进入 shortlist",
    )
    shortlist_parser.add_argument(
        "--max-blocked-processes",
        type=int,
        default=0,
        help="最多允许的 blocked process_alignment 过程数，默认不接受 blocked process",
    )
    shortlist_parser.add_argument(
        "--min-process-alignment-score",
        type=float,
        default=None,
        help="最小 process alignment overall score",
    )
    shortlist_parser.add_argument("--top", type=int, default=5, help="返回前 N 篇")
    shortlist_parser.add_argument(
        "--export", type=str, help="导出 Markdown shortlist 文件"
    )

    # submission-board 命令
    board_parser = subparsers.add_parser(
        "submission-board", help="按 venue 查看当前最值得投稿的论文"
    )
    board_parser.add_argument(
        "--top", type=int, default=3, help="每个 venue 显示前 N 篇"
    )
    board_parser.add_argument(
        "--min-breakthrough",
        type=float,
        default=None,
        help="只显示突破潜力高于阈值的稿件",
    )
    board_parser.add_argument(
        "--min-priority",
        type=float,
        default=None,
        help="只显示投稿优先级高于阈值的稿件",
    )
    board_parser.add_argument(
        "--max-blockers",
        type=int,
        default=None,
        help="只显示 blocker 数不超过阈值的稿件",
    )
    board_parser.add_argument(
        "--min-rewrite-gain",
        type=float,
        default=None,
        help="只显示 rewrite priority gain 高于阈值的稿件",
    )
    board_parser.add_argument(
        "--require-gate", action="store_true", help="只显示通过质量门槛的稿件"
    )
    board_parser.add_argument(
        "--max-fallbacks",
        type=int,
        default=None,
        help="最多允许的 fallback 事件数",
    )
    board_parser.add_argument(
        "--max-strict-fallbacks",
        type=int,
        default=0,
        help="最多允许的 strict fallback 事件数，默认不接受 strict fallback",
    )
    board_parser.add_argument(
        "--max-blocked-stages",
        type=int,
        default=0,
        help="最多允许的 blocked stage standards 数，默认不接受 blocked stage",
    )
    board_parser.add_argument(
        "--max-missing-stages",
        type=int,
        default=None,
        help="最多允许的 missing stage standards 数",
    )
    board_parser.add_argument(
        "--max-attention-stages",
        type=int,
        default=None,
        help="最多允许的 needs_attention stage standards 数",
    )
    board_parser.add_argument(
        "--min-stage-score",
        type=float,
        default=None,
        help="最小 stage standards overall score",
    )
    board_parser.add_argument(
        "--max-self-evolution-required-failures",
        type=int,
        default=0,
        help="最多允许的 self-evolution required failure 数，默认不接受 required failure",
    )
    board_parser.add_argument(
        "--min-self-evolution-score",
        type=float,
        default=None,
        help="最小 self-evolution score",
    )
    board_parser.add_argument(
        "--allow-blocked-self-evolution",
        action="store_true",
        help="允许 self-evolution status 为 blocked 的稿件进入投稿看板",
    )
    board_parser.add_argument(
        "--max-blocked-processes",
        type=int,
        default=0,
        help="最多允许的 blocked process_alignment 过程数，默认不接受 blocked process",
    )
    board_parser.add_argument(
        "--min-process-alignment-score",
        type=float,
        default=None,
        help="最小 process alignment overall score",
    )
    board_parser.add_argument("--export", type=str, help="导出 Markdown board 文件")

    # rewrite-board 命令
    rewrite_board_parser = subparsers.add_parser(
        "rewrite-board", help="查看最值得继续重写优化的论文"
    )
    rewrite_board_parser.add_argument(
        "--type", choices=["icbinb", "normal", "journal", "extended"], help="按类型过滤"
    )
    rewrite_board_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        help="按目标 venue 过滤",
    )
    rewrite_board_parser.add_argument(
        "--min-priority",
        type=float,
        default=None,
        help="只显示投稿优先级高于阈值的稿件",
    )
    rewrite_board_parser.add_argument(
        "--min-rewrite-gain",
        type=float,
        default=None,
        help="只显示 rewrite priority gain 高于阈值的稿件",
    )
    rewrite_board_parser.add_argument(
        "--max-blockers",
        type=int,
        default=None,
        help="只显示 blocker 数不超过阈值的稿件",
    )
    rewrite_board_parser.add_argument(
        "--require-gate", action="store_true", help="只显示通过质量门槛的稿件"
    )
    rewrite_board_parser.add_argument(
        "--include-ready", action="store_true", help="包含已 ready 的稿件"
    )
    rewrite_board_parser.add_argument("--top", type=int, default=10, help="返回前 N 篇")
    rewrite_board_parser.add_argument(
        "--export", type=str, help="导出 Markdown board 文件"
    )

    repair_board_parser = subparsers.add_parser(
        "repair-board", help="查看 reviewer 反馈转成的结构化修复任务"
    )
    repair_board_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        help="按目标 venue 过滤",
    )
    repair_board_parser.add_argument(
        "--priority-tier",
        choices=["p0", "p1", "p2"],
        help="只显示指定优先级 tier 的修复任务",
    )
    repair_board_parser.add_argument(
        "--only-ready",
        action="store_true",
        help="只显示已经具备动作和验证计划的修复任务",
    )
    repair_board_parser.add_argument(
        "--top", type=int, default=20, help="返回前 N 个 repair tasks"
    )
    repair_board_parser.add_argument(
        "--export", type=str, help="导出 Markdown repair board 文件"
    )

    evolution_board_parser = subparsers.add_parser(
        "evolution-board",
        help="查看 reviewer 修复闭环沉淀出的 self-evolution 结果",
    )
    evolution_board_parser.add_argument(
        "--status",
        choices=["ready", "needs_attention", "blocked"],
        help="按 self-evolution 状态过滤",
    )
    evolution_board_parser.add_argument(
        "--top", type=int, default=20, help="返回前 N 个 self-evolution 项目"
    )

    # submission-dossier 命令
    dossier_parser = subparsers.add_parser(
        "submission-dossier", help="导出单篇论文的投稿材料包"
    )
    dossier_parser.add_argument("folder", help="论文文件夹名称")
    dossier_parser.add_argument("output_dir", help="导出目录")

    benchmark_parser = subparsers.add_parser(
        "readiness-benchmark",
        help="汇总当前研究产出的投稿 readiness benchmark",
    )
    benchmark_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        default="nature",
        help="按目标 venue 的门槛做基准评估",
    )
    benchmark_parser.add_argument(
        "--top", type=int, default=10, help="展示前 N 篇 benchmark 结果"
    )
    benchmark_parser.add_argument(
        "--max-entries",
        type=int,
        default=200,
        help="最多扫描多少篇历史结果",
    )
    benchmark_parser.add_argument(
        "--include-other-venues",
        action="store_true",
        help="包含 target venue 不匹配的稿件作为横向参考",
    )
    benchmark_parser.add_argument(
        "--export", type=str, help="导出 Markdown benchmark 文件"
    )

    pipeline_status_parser = subparsers.add_parser(
        "pipeline-status",
        help="查看 contracts 驱动的项目流水线状态",
    )
    pipeline_status_parser.add_argument(
        "--top", type=int, default=20, help="显示前 N 个项目"
    )

    stage_standards_parser = subparsers.add_parser(
        "stage-standards",
        help="查看每个流程阶段的结构化评估标准与得分",
    )
    stage_standards_parser.add_argument(
        "--top", type=int, default=60, help="显示前 N 条阶段记录"
    )
    stage_standards_parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="按阶段过滤，如 ideation / planning / experiment / figure / manuscript / review",
    )
    stage_standards_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按阶段状态过滤，如 ready / blocked / needs_attention / missing",
    )

    process_board_parser = subparsers.add_parser(
        "process-board",
        help="查看点对点对标开源参考的过程级对齐状态",
    )
    process_board_parser.add_argument(
        "--top", type=int, default=80, help="显示前 N 条过程记录"
    )
    process_board_parser.add_argument(
        "--process",
        type=str,
        default=None,
        help="按过程过滤，如 ideation / program / exploration / experiment / figure / writing / review / evolution / packaging",
    )
    process_board_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按过程状态过滤，如 ready / blocked / needs_attention / missing",
    )

    fallback_board_parser = subparsers.add_parser(
        "fallback-board",
        help="查看 pipeline fallback 事件与兜底债务",
    )
    fallback_board_parser.add_argument(
        "--top", type=int, default=30, help="显示前 N 个项目"
    )
    fallback_board_parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="按 fallback stage 过滤",
    )

    idea_board_parser = subparsers.add_parser(
        "idea-board",
        help="查看结构化 idea card 看板",
    )
    idea_board_parser.add_argument(
        "--top", type=int, default=30, help="显示前 N 条 idea"
    )
    idea_board_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按 idea status 过滤",
    )

    experiment_board_parser = subparsers.add_parser(
        "experiment-board",
        help="查看 experiment registry 看板",
    )
    experiment_board_parser.add_argument(
        "--top", type=int, default=50, help="显示前 N 条实验记录"
    )
    experiment_board_parser.add_argument(
        "--status",
        type=str,
        default=None,
        help="按实验状态过滤",
    )

    figure_board_parser = subparsers.add_parser(
        "figure-board",
        help="查看 figure spec 看板",
    )
    figure_board_parser.add_argument(
        "--top", type=int, default=50, help="显示前 N 条 figure 记录"
    )
    figure_board_parser.add_argument(
        "--ready-only",
        action="store_true",
        help="只显示 ready 的 figure",
    )

    source_board_parser = subparsers.add_parser(
        "source-board",
        help="查看 source lineage 与 planning 看板",
    )
    source_board_parser.add_argument(
        "--top", type=int, default=30, help="显示前 N 个 source"
    )
    source_board_parser.add_argument(
        "--archetype",
        type=str,
        default=None,
        help="按 source archetype 过滤",
    )

    source_mix_parser = subparsers.add_parser(
        "source-mix",
        help="查看 source mix 与下一批研究倾斜建议",
    )
    source_mix_parser.add_argument(
        "--desired-policy",
        type=str,
        default=None,
        help="按当前期望 workflow / execution policy 给建议",
    )
    source_mix_parser.add_argument(
        "--top", type=int, default=50, help="最多纳入多少个 source 做分析"
    )

    source_next_batch_parser = subparsers.add_parser(
        "source-next-batch",
        help="生成下一批 source 组合与节奏建议",
    )
    source_next_batch_parser.add_argument(
        "--desired-policy",
        type=str,
        default=None,
        help="按当前期望 workflow / execution policy 生成组合建议",
    )
    source_next_batch_parser.add_argument(
        "--top", type=int, default=50, help="最多纳入多少个 source 做分析"
    )
    source_next_batch_parser.add_argument(
        "--max-slots",
        type=int,
        default=3,
        help="最多输出多少个下一批组合 lane",
    )

    trend_parser = subparsers.add_parser(
        "benchmark-trends",
        help="按日期查看 readiness benchmark 趋势",
    )
    trend_parser.add_argument(
        "--venue",
        choices=["neurips", "iclr", "cvpr", "journal", "nature"],
        default="nature",
        help="趋势统计的目标 venue",
    )
    trend_parser.add_argument(
        "--max-entries",
        type=int,
        default=200,
        help="最多扫描多少篇历史结果",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    manager = ResearchManager(args.research_dir)

    if args.command == "list-batches":
        batches = manager.list_batches()
        print(f"\n共有 {len(batches)} 个批次:\n")
        for batch in batches:
            completed = len(batch["progress"].get("papers_completed", []))
            failed = len(batch["progress"].get("papers_failed", []))
            print(f"📁 {batch['name']}")
            print(f"   创建时间: {batch['created_at']}")
            print(f"   状态: {completed} 完成, {failed} 失败")
            print()

    elif args.command == "batch-summary":
        summary = manager.get_batch_summary(args.batch_name)
        if not summary:
            print(f"❌ 未找到批次: {args.batch_name}")
            sys.exit(1)

        print(f"\n批次: {summary['batch_name']}")
        print(f"路径: {summary['path']}")
        source_provenance = (
            summary.get("report", {}).get("source_provenance")
            or summary.get("progress", {}).get("source_provenance")
            or {}
        )
        if source_provenance:
            print(
                "来源: "
                f"{source_provenance.get('source_name') or source_provenance.get('source_key')} | "
                f"workflow={source_provenance.get('source_workflow_mode')} | "
                f"archetype={source_provenance.get('source_archetype')} | "
                f"profile={source_provenance.get('source_batch_profile')}"
            )
        print(f"\n论文 ({len(summary['papers'])} 个):")
        for paper in summary["papers"]:
            print(f"  📄 {paper['name']} ({paper['type']})")
            print(f"     {paper['path']}")

        if summary["report"]:
            stats = summary["report"].get("statistics", {})
            print(f"\n统计:")
            print(f"  总计: {stats.get('total_papers', 0)}")
            print(f"  成功: {stats.get('completed', 0)}")
            print(f"  失败: {stats.get('failed', 0)}")

            quality_summary = summary["report"].get("quality_summary", {})
            if quality_summary:
                if quality_summary.get("avg_quality_score") is not None:
                    print(
                        f"  平均质量分: {quality_summary.get('avg_quality_score'):.2f}"
                    )
                if quality_summary.get("avg_rigor_score") is not None:
                    print(f"  平均严谨性: {quality_summary.get('avg_rigor_score'):.2f}")
                print(f"  质量门槛通过: {quality_summary.get('gate_passed', 0)}")
                print(f"  质量门槛未过: {quality_summary.get('gate_failed', 0)}")
                top_papers = quality_summary.get("top_papers", [])
                if top_papers:
                    print(f"\n推荐优先查看:")
                    for paper in top_papers[:3]:
                        print(
                            f"  - idea #{paper.get('idea_idx')} [{paper.get('paper_type')}] "
                            f"quality={paper.get('quality_score')} rigor={paper.get('rigor_score')} "
                            f"gate={paper.get('quality_gate_passed')}"
                        )

        failure_summary = summary.get("failure_summary", {})
        if failure_summary.get("total", 0):
            print(f"\n失败诊断:")
            for stage, count in sorted(failure_summary.get("by_stage", {}).items()):
                print(f"  {stage}: {count}")
            for sample in failure_summary.get("samples", []):
                print(
                    f"  - idea #{sample.get('idea_idx')} [{sample.get('paper_type')}] "
                    f"stage={sample.get('stage')} error={sample.get('error', '')}"
                )

    elif args.command == "list-papers":
        papers = manager.list_papers(args.type, args.sort)
        print(f"\n共有 {len(papers)} 篇论文:\n")
        for paper in papers:
            print(f"📄 {paper['name']}")
            print(f"   类型: {paper['type']}")
            print(f"   文件夹: {paper['folder']}")
            print(f"   大小: {format_size(paper['size'])}")
            print(f"   创建时间: {paper['created_at']}")
            if paper.get("latest_stage"):
                print(f"   阶段: {paper['latest_stage']}")
            if paper.get("has_reviews") is not None:
                print(f"   审查: {'yes' if paper['has_reviews'] else 'no'}")
            if isinstance(paper.get("quality_score"), (int, float)):
                print(f"   质量分: {paper['quality_score']:.2f}")
            if isinstance(paper.get("rigor_score"), (int, float)):
                print(f"   严谨性: {paper['rigor_score']:.2f}")
            if isinstance(paper.get("claim_support_score"), (int, float)):
                print(f"   论证支撑: {paper['claim_support_score']:.2f}")
            if isinstance(paper.get("claim_alignment_score"), (int, float)):
                print(f"   论断对齐: {paper['claim_alignment_score']:.2f}")
            if isinstance(paper.get("numeric_coverage_score"), (int, float)):
                print(f"   数值覆盖: {paper['numeric_coverage_score']:.2f}")
            if isinstance(paper.get("breakthrough_score"), (int, float)):
                print(f"   突破潜力: {paper['breakthrough_score']:.2f}")
            if paper.get("claims_detected") is not None:
                print(f"   claims: {paper['claims_detected']}")
            if paper.get("unsupported_claims_count") is not None:
                print(f"   unsupported claims: {paper['unsupported_claims_count']}")
            if paper.get("suggested_claim_rewrites_count") is not None:
                print(
                    f"   suggested claim rewrites: {paper['suggested_claim_rewrites_count']}"
                )
            if paper.get("num_figures") is not None:
                print(
                    f"   figures/tables: {paper['num_figures']}/{paper.get('num_tables', 0)}"
                )
            if isinstance(paper.get("evidence_density_score"), (int, float)):
                print(f"   证据密度: {paper['evidence_density_score']:.2f}")
            if paper.get("key_results_count") is not None:
                print(f"   key results: {paper['key_results_count']}")
            if paper.get("structured_results_count") is not None:
                print(f"   structured results: {paper['structured_results_count']}")
            if paper.get("contribution_count") is not None:
                print(f"   contributions: {paper['contribution_count']}")
            if paper.get("target_venue"):
                print(f"   目标 venue: {paper['target_venue']}")
            if paper.get("submission_status"):
                print(f"   投稿准备度: {paper['submission_status']}")
            if isinstance(paper.get("submission_priority_score"), (int, float)):
                print(
                    f"   投稿优先级: {paper['submission_priority_score']:.2f} ({paper.get('submission_priority_tier')})"
                )
            if isinstance(paper.get("blocker_count"), int):
                print(f"   blocker 数: {paper['blocker_count']}")
            if isinstance(paper.get("experiment_todo_count"), int):
                print(
                    f"   experiment TODO: total={paper.get('experiment_todo_count')} p0={paper.get('experiment_todo_p0_count')}"
                )
            if isinstance(paper.get("experiment_todo_closure_rate"), (int, float)):
                print(
                    f"   experiment TODO closure: total={paper.get('experiment_todo_closure_rate'):.2f} p0={paper.get('experiment_todo_p0_closure_rate')}"
                )
            if isinstance(paper.get("experiment_todo_unresolved_count"), int):
                print(
                    f"   experiment TODO unresolved/closed: {paper.get('experiment_todo_unresolved_count')}/{paper.get('experiment_todo_closed_count')}"
                )
            if paper.get("experiment_todo_top_action"):
                print(f"   experiment TODO top: {paper['experiment_todo_top_action']}")
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
                print(f"   rewrite 增益: {paper['rewrite_priority_gain_total']:.2f}")
            if paper.get("experiment_todo_file") and args.detailed:
                print(f"   experiment TODO file: {paper['experiment_todo_file']}")
            if paper.get("experiment_todo_progress_file") and args.detailed:
                print(
                    f"   experiment TODO progress: {paper['experiment_todo_progress_file']}"
                )
            if paper.get("submission_package_file") and args.detailed:
                print(f"   submission package: {paper['submission_package_file']}")
            if paper.get("narrative_map_file") and args.detailed:
                print(f"   narrative map: {paper['narrative_map_file']}")
            if paper.get("result_story_file") and args.detailed:
                print(f"   result story: {paper['result_story_file']}")
            if paper.get("contribution_map_file") and args.detailed:
                print(f"   contribution map: {paper['contribution_map_file']}")
            if paper.get("editor_pitch_file") and args.detailed:
                print(f"   editor pitch: {paper['editor_pitch_file']}")
            if paper.get("rebuttal_package_file") and args.detailed:
                print(f"   rebuttal package: {paper['rebuttal_package_file']}")
            if paper.get("risk_register_file") and args.detailed:
                print(f"   risk register: {paper['risk_register_file']}")
            if paper.get("cover_letter_file") and args.detailed:
                print(f"   cover letter: {paper['cover_letter_file']}")
            if paper.get("abstract_polish_file") and args.detailed:
                print(f"   abstract polish: {paper['abstract_polish_file']}")
            if paper.get("impact_brief_file") and args.detailed:
                print(f"   impact brief: {paper['impact_brief_file']}")
            if paper.get("contribution_bullets_file") and args.detailed:
                print(f"   contribution bullets: {paper['contribution_bullets_file']}")
            if paper.get("strongest_claims_file") and args.detailed:
                print(f"   strongest claims: {paper['strongest_claims_file']}")
            if paper.get("submission_manifest_file") and args.detailed:
                print(f"   submission manifest: {paper['submission_manifest_file']}")
            if paper.get("submission_dashboard_file") and args.detailed:
                print(f"   submission dashboard: {paper['submission_dashboard_file']}")
            if paper.get("risk_language_plan_file") and args.detailed:
                print(f"   risk language plan: {paper['risk_language_plan_file']}")
            if paper.get("claim_softening_plan_file") and args.detailed:
                print(f"   claim softening plan: {paper['claim_softening_plan_file']}")
            if paper.get("rewrite_effectiveness_file") and args.detailed:
                print(
                    f"   rewrite effectiveness: {paper['rewrite_effectiveness_file']}"
                )
            if paper.get("rewrite_trace_summary_file") and args.detailed:
                print(
                    f"   rewrite trace summary: {paper['rewrite_trace_summary_file']}"
                )
            if isinstance(paper.get("rewrite_round_count"), int):
                print(f"   rewrite rounds: {paper['rewrite_round_count']}")
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
                print(
                    f"   rewrite priority gain: {paper['rewrite_priority_gain_total']:.2f}"
                )
            if isinstance(paper.get("rewrite_best_round_priority_delta"), (int, float)):
                print(
                    f"   best rewrite round delta: {paper['rewrite_best_round_priority_delta']:.2f}"
                )
            if paper.get("rewrite_top_section"):
                print(f"   top rewrite section: {paper['rewrite_top_section']}")
            if paper.get("quality_rewrite_applied") is not None:
                print(
                    f"   高质量重写: {'yes' if paper['quality_rewrite_applied'] else 'no'}"
                )
            if paper.get("quality_gate_passed") is not None:
                print(
                    f"   质量门槛: {'pass' if paper['quality_gate_passed'] else 'fail'}"
                )
            if paper.get("quality_status"):
                print(f"   质量状态: {paper['quality_status']}")
            if args.detailed:
                print(f"   路径: {paper['path']}")
            print()

    elif args.command == "paper-details":
        details = manager.get_paper_details(args.folder)
        if not details:
            print(f"❌ 未找到论文文件夹: {args.folder}")
            sys.exit(1)

        print(f"\n论文详细信息: {details['folder']}")
        print(f"路径: {details['path']}\n")

        if details["idea"]:
            print("📝 想法信息:")
            print(f"   名称: {details['idea'].get('Name', 'N/A')}")
            print(f"   标题: {details['idea'].get('Title', 'N/A')}")
            print(f"   摘要: {details['idea'].get('Abstract', 'N/A')[:200]}...")
            print()

        print("📁 文件列表:")
        for file_info in details["files"]:
            print(f"   {file_info['name']} ({format_size(file_info['size'])})")

        if details["reviews"]:
            print(f"\n🔍 审查记录 ({len(details['reviews'])} 轮):")
            for review in details["reviews"]:
                print(f"   {review['round']}")

        if details.get("quality"):
            quality = details["quality"]
            print(f"\n🏁 高质量摘要:")
            print(f"   target venue: {quality.get('target_venue')}")
            print(f"   quality after: {quality.get('quality_score_after')}")
            print(f"   rigor after: {quality.get('rigor_score_after')}")
            print(f"   claim support after: {quality.get('claim_support_after')}")
            print(f"   claim alignment after: {quality.get('claim_alignment_after')}")
            print(f"   numeric coverage after: {quality.get('numeric_coverage_after')}")
            print(f"   breakthrough score: {quality.get('breakthrough_score')}")
            print(f"   evidence density: {quality.get('evidence_density_score')}")
            print(
                f"   submission priority: {quality.get('submission_priority_score')} ({quality.get('submission_priority_tier')})"
            )
            print(f"   gate passed: {quality.get('quality_gate_passed')}")
            readiness = quality.get("submission_readiness", {})
            if readiness:
                print(f"   readiness: {readiness.get('status')}")
                for blocker in readiness.get("blockers", [])[:5]:
                    print(f"     - {blocker}")
                categories = readiness.get("categories", {})
                if categories:
                    print(f"   blocker categories: {categories}")
            unsupported = quality.get("unsupported_claims_count")
            if unsupported is not None:
                print(f"   unsupported claims: {unsupported}")
            claim_rewrites = quality.get("suggested_claim_rewrites_count")
            if claim_rewrites is not None:
                print(f"   suggested claim rewrites: {claim_rewrites}")
            revision_actions = quality.get("revision_actions") or []
            if revision_actions:
                print("   revision actions:")
                for item in revision_actions[:4]:
                    print(
                        f"     - [{item.get('priority')}] {item.get('focus')}: {item.get('action')}"
                    )

        if details.get("evidence_pack"):
            evidence = details["evidence_pack"]
            print(f"\n📎 Strongest Results:")
            for item in evidence.get("strongest_results", [])[:3]:
                print(
                    f"   - {item.get('type')}:{item.get('label')} "
                    f"refs={item.get('ref_count')} caption={item.get('caption', '')[:120]}"
                )

        if details.get("key_results"):
            print(f"\n🔢 Key Numerical Results:")
            for value in details["key_results"].get("values", [])[:10]:
                print(f"   - {value}")

        if details.get("claim_alignment"):
            print(f"\n🔗 Claim Alignment:")
            for item in details["claim_alignment"].get("claims", [])[:5]:
                if item.get("suggested_rewrite"):
                    print(f"   - claim: {item.get('claim')[:120]}")
                    print(f"     suggestion: {item.get('suggested_rewrite')}")

        if details.get("contribution_map"):
            print(f"\n🧭 Contribution Map:")
            for item in details["contribution_map"].get("contributions", [])[:3]:
                print(f"   - {item.get('title')}: {item.get('claim')[:120]}")
                print(
                    f"     evidence: {', '.join(item.get('evidence_labels', [])) or 'n/a'}"
                )
                print(
                    f"     key results: {', '.join(item.get('key_results', [])) or 'n/a'}"
                )

        if details.get("submission_dashboard"):
            print("\n🧪 Submission Dashboard:\n")
            print(details["submission_dashboard"][:1200])

        if details.get("editor_pitch"):
            print("\n📝 Editor Pitch:\n")
            print(details["editor_pitch"][:1200])

        if details.get("narrative_map"):
            print("\n🗺️ Narrative Map:\n")
            print(details["narrative_map"][:1200])

        if details.get("result_story"):
            print("\n📚 Result Story:\n")
            print(details["result_story"][:1200])

        if details.get("impact_brief"):
            print("\n🌍 Impact Brief:\n")
            print(details["impact_brief"][:1200])

        if details.get("contribution_bullets"):
            print("\n📌 Contribution Bullets:\n")
            print(details["contribution_bullets"][:1200])

        if details.get("strongest_claims"):
            print("\n💥 Strongest Claims:\n")
            print(details["strongest_claims"][:1200])

        if details.get("risk_register"):
            print("\n⚠️ Risk Register:\n")
            print(details["risk_register"][:1200])

        if details.get("cover_letter"):
            print("\n📨 Cover Letter:\n")
            print(details["cover_letter"][:1200])

        if details.get("abstract_polish"):
            print("\n✍️ Abstract Polish:\n")
            print(details["abstract_polish"][:1200])

        if details.get("rebuttal_package"):
            print("\n🛡️ Rebuttal Package:\n")
            print(details["rebuttal_package"][:1200])

        if details.get("risk_language_plan"):
            print("\n🧷 Risk Language Plan:\n")
            print(details["risk_language_plan"][:1200])

        if details.get("claim_softening_plan"):
            print("\n🪶 Claim Softening Plan:\n")
            print(details["claim_softening_plan"][:1200])

        if details.get("rewrite_effectiveness"):
            print("\n📈 Rewrite Effectiveness:\n")
            print(details["rewrite_effectiveness"][:1200])

        if details.get("rewrite_trace_summary"):
            trace = details["rewrite_trace_summary"]
            print("\n🧮 Rewrite Trace Summary:\n")
            print(f"  rounds: {trace.get('round_count')}")
            print(f"  total priority gain: {trace.get('priority_gain_total')}")
            print(
                f"  avg priority gain / round: {trace.get('avg_priority_gain_per_round')}"
            )
            best_round = trace.get("best_round") or {}
            if best_round:
                print(
                    f"  best round: {best_round.get('round')} (delta={best_round.get('priority_delta')}, quality={best_round.get('quality_delta')})"
                )
            if trace.get("top_frontmatter_style"):
                print(f"  top frontmatter style: {trace.get('top_frontmatter_style')}")
            if trace.get("top_section_style"):
                print(f"  top section style: {trace.get('top_section_style')}")
            if trace.get("top_section"):
                print(f"  top section: {trace.get('top_section')}")
    elif args.command == "list-ideas":
        ideas = manager.list_ideas()
        print(f"\n共有 {len(ideas)} 个想法:\n")
        for idea in ideas:
            print(f"💡 {idea['name']}")
            if idea["title"]:
                print(f"   标题: {idea['title']}")
            print(f"   来源: {idea['source']}")
            print()

    elif args.command == "search-papers":
        papers = manager.search_papers(args.query, args.type)
        print(f"\n找到 {len(papers)} 篇匹配 '{args.query}' 的论文:\n")
        for paper in papers:
            print(f"📄 {paper['name']} ({paper['type']})")
            print(f"   {paper['path']}")
            print()

    elif args.command == "cleanup":
        manager.cleanup_old_files(args.days, args.dry_run)

    elif args.command == "stats":
        batches = manager.list_batches()
        papers = manager.list_papers()
        ideas = manager.list_ideas()
        index_summary = manager.get_index_summary()
        failure_stage_counts = Counter()
        quality_scores = []
        claim_support_scores = []
        priority_scores = []
        rewrite_gains = []
        rewrite_top_sections = Counter()
        rewrite_top_frontmatter_styles = Counter()
        priority_tiers = Counter()
        quality_gate_pass = 0
        quality_gate_fail = 0
        for batch in batches:
            for failed in batch.get("progress", {}).get("papers_failed", []):
                failure_stage_counts[failed.get("stage") or "unknown"] += 1
        for paper in papers:
            if isinstance(paper.get("quality_score"), (int, float)):
                quality_scores.append(paper["quality_score"])
            if isinstance(paper.get("claim_support_score"), (int, float)):
                claim_support_scores.append(paper["claim_support_score"])
            if isinstance(paper.get("submission_priority_score"), (int, float)):
                priority_scores.append(paper["submission_priority_score"])
            if paper.get("submission_priority_tier"):
                priority_tiers[paper["submission_priority_tier"]] += 1
            if isinstance(paper.get("rewrite_priority_gain_total"), (int, float)):
                rewrite_gains.append(paper["rewrite_priority_gain_total"])
            if paper.get("rewrite_top_section"):
                rewrite_top_sections[paper["rewrite_top_section"]] += 1
            if paper.get("rewrite_top_frontmatter_style"):
                rewrite_top_frontmatter_styles[
                    paper["rewrite_top_frontmatter_style"]
                ] += 1
            if paper.get("quality_gate_passed") is True:
                quality_gate_pass += 1
            elif paper.get("quality_gate_passed") is False:
                quality_gate_fail += 1

        print(f"\n📊 研究目录统计")
        print(f"=" * 50)
        print(f"目录: {args.research_dir}")
        print()
        print(f"批次: {len(batches)}")
        print(f"论文: {len(papers)}")
        print(f"想法: {len(ideas)}")
        print(f"索引条目: {index_summary['entries']}")
        print()

        # 按类型统计论文
        paper_types = {}
        for paper in papers:
            paper_type = paper["type"]
            paper_types[paper_type] = paper_types.get(paper_type, 0) + 1

        print("按类型统计:")
        for paper_type, count in sorted(paper_types.items()):
            print(f"  {paper_type}: {count}")

        if index_summary["generated_at"]:
            print()
            print(f"索引更新时间: {index_summary['generated_at']}")
            for category, count in sorted(index_summary["by_category"].items()):
                print(f"  index/{category}: {count}")

        if quality_scores:
            print()
            print(f"平均质量分: {sum(quality_scores) / len(quality_scores):.2f}")
            print(f"最高质量分: {max(quality_scores):.2f}")
            if claim_support_scores:
                print(
                    f"平均论证支撑: {sum(claim_support_scores) / len(claim_support_scores):.2f}"
                )
            print(f"质量门槛通过: {quality_gate_pass}")
            print(f"质量门槛未过: {quality_gate_fail}")
            if priority_scores:
                print(
                    f"平均投稿优先级: {sum(priority_scores) / len(priority_scores):.2f}"
                )
                if rewrite_gains:
                    print(
                        f"平均 rewrite 增益: {sum(rewrite_gains) / len(rewrite_gains):.2f}"
                    )
                for tier, count in sorted(priority_tiers.items()):
                    print(f"  priority/{tier}: {count}")
                if rewrite_top_sections:
                    print("  热门重写章节:")
                    for section, count in rewrite_top_sections.most_common(5):
                        print(f"    - {section}: {count}")
                if rewrite_top_frontmatter_styles:
                    print("  热门 frontmatter 风格:")
                    for style, count in rewrite_top_frontmatter_styles.most_common(5):
                        print(f"    - {style}: {count}")

        if failure_stage_counts:
            print()
            print("失败阶段摘要:")
            for stage, count in sorted(failure_stage_counts.items()):
                print(f"  {stage}: {count}")

    elif args.command == "rebuild-index":
        index = manager.rebuild_index()
        print(f"✅ 索引已重建: {run_index_path(args.research_dir)}")
        print(f"条目数: {len(index.get('entries', {}))}")

    elif args.command == "shortlist":
        shortlist = manager.shortlist_papers(
            paper_type=args.type,
            target_venue=args.venue,
            require_gate=args.require_gate,
            require_ready=args.require_ready,
            min_breakthrough=args.min_breakthrough,
            min_priority=args.min_priority,
            max_blockers=args.max_blockers,
            min_rewrite_gain=args.min_rewrite_gain,
            max_fallbacks=args.max_fallbacks,
            max_strict_fallbacks=args.max_strict_fallbacks,
            max_blocked_stages=args.max_blocked_stages,
            max_missing_stages=args.max_missing_stages,
            max_attention_stages=args.max_attention_stages,
            min_stage_score=args.min_stage_score,
            max_self_evolution_required_failures=args.max_self_evolution_required_failures,
            min_self_evolution_score=args.min_self_evolution_score,
            allow_blocked_self_evolution=args.allow_blocked_self_evolution,
            max_blocked_processes=args.max_blocked_processes,
            min_process_alignment_score=args.min_process_alignment_score,
            top_n=args.top,
        )
        print(f"\n🎯 投稿 shortlist ({len(shortlist)} 篇):\n")
        for paper in shortlist:
            print(f"📄 {paper['name']}")
            print(f"   类型: {paper['type']}")
            print(f"   venue: {paper.get('target_venue')}")
            print(
                f"   投稿优先级: {paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')})"
            )
            print(f"   rewrite 增益: {paper.get('rewrite_priority_gain_total')}")
            print(f"   blocker 数: {paper.get('blocker_count')}")
            print(
                f"   阶段标准: score={paper.get('stage_overall_score')} blocked={paper.get('blocked_stage_count')} "
                f"attention={paper.get('needs_attention_stage_count')} missing={paper.get('missing_stage_count')}"
            )
            print(
                f"   reviewer修复: resolution={paper.get('review_resolution_rate')} active={paper.get('review_active_issue_count')} "
                f"persistent={paper.get('review_persistent_issue_count')} checks={paper.get('review_verification_count')}"
            )
            print(
                f"   reviewer绑定: coverage={paper.get('review_target_binding_coverage')} active_coverage={paper.get('review_active_binding_coverage')} "
                f"unbound={paper.get('review_unbound_issue_count')}"
            )
            print(
                f"   reviewer修复队列: queue={paper.get('review_repair_queue_count')} ready={paper.get('review_repair_ready_count')} "
                f"ready_coverage={paper.get('review_repair_ready_coverage')} verification_ready={paper.get('review_repair_verification_ready_count')}"
            )
            print(
                f"   self-evolution: status={paper.get('self_evolution_status')} score={paper.get('self_evolution_score')} "
                f"required_failures={paper.get('self_evolution_required_failure_count')} lane={paper.get('self_evolution_dominant_lane')} "
                f"role={paper.get('self_evolution_dominant_role')}"
            )
            if paper.get("self_evolution_top_risks"):
                print(
                    "   self-evolution风险: "
                    + ", ".join(paper.get("self_evolution_top_risks") or [])
                )
            print(
                f"   process-alignment: score={paper.get('process_alignment_overall_score')} "
                f"blocked={paper.get('process_alignment_blocked_process_count')} attention={paper.get('process_alignment_attention_process_count')} "
                f"missing={paper.get('process_alignment_missing_process_count')}"
            )
            if paper.get("process_alignment_top_risks"):
                print(
                    "   process风险: "
                    + ", ".join(paper.get("process_alignment_top_risks") or [])
                )
            print(
                f"   fallback: total={paper.get('fallback_count')} strict={paper.get('strict_fallback_count')}"
            )
            print(f"   质量分: {paper.get('quality_score')}")
            print(f"   严谨性: {paper.get('rigor_score')}")
            print(f"   论证支撑: {paper.get('claim_support_score')}")
            print(f"   门槛: {paper.get('quality_gate_passed')}")
            print(f"   路径: {paper['path']}")
            print()

        if args.export:
            export_path = manager.export_shortlist_markdown(shortlist, args.export)
            print(f"📝 已导出 shortlist: {export_path}")

    elif args.command == "submission-board":
        board = manager.submission_board(
            args.top,
            args.min_breakthrough,
            args.min_priority,
            args.max_blockers,
            args.min_rewrite_gain,
            args.require_gate,
            args.max_fallbacks,
            args.max_strict_fallbacks,
            args.max_blocked_stages,
            args.max_missing_stages,
            args.max_attention_stages,
            args.min_stage_score,
            args.max_self_evolution_required_failures,
            args.min_self_evolution_score,
            args.allow_blocked_self_evolution,
            args.max_blocked_processes,
            args.min_process_alignment_score,
        )
        print("\n🗂️ Submission Board\n")
        for venue, papers in sorted(board.items()):
            print(f"## {venue}")
            for paper in papers:
                print(
                    f"- {paper['name']} | priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                    f"rewrite_gain={paper.get('rewrite_priority_gain_total')} | blockers={paper.get('blocker_count')} | "
                    f"stage_score={paper.get('stage_overall_score')} blocked_stages={paper.get('blocked_stage_count')} "
                    f"attention_stages={paper.get('needs_attention_stage_count')} missing_stages={paper.get('missing_stage_count')} | "
                    f"self_evolution={paper.get('self_evolution_status')}:{paper.get('self_evolution_score')} "
                    f"required_failures={paper.get('self_evolution_required_failure_count')} | "
                    f"process_alignment={paper.get('process_alignment_overall_score')} "
                    f"blocked_processes={paper.get('process_alignment_blocked_process_count')} | "
                    f"review_resolution={paper.get('review_resolution_rate')} review_binding={paper.get('review_target_binding_coverage')} "
                    f"repair_ready={paper.get('review_repair_ready_coverage')} "
                    f"active_review_issues={paper.get('review_active_issue_count')} persistent_review_issues={paper.get('review_persistent_issue_count')} | "
                    f"fallbacks={paper.get('fallback_count')} strict={paper.get('strict_fallback_count')} | breakthrough={paper.get('breakthrough_score')} | "
                    f"rigor={paper.get('rigor_score')} | gate={paper.get('quality_gate_passed')}"
                )
            print()

        if args.export:
            export_path = manager.export_submission_board_markdown(board, args.export)
            print(f"📝 已导出 submission board: {export_path}")

    elif args.command == "rewrite-board":
        papers = manager.rewrite_board(
            top_n=args.top,
            paper_type=args.type,
            target_venue=args.venue,
            min_priority=args.min_priority,
            min_rewrite_gain=args.min_rewrite_gain,
            max_blockers=args.max_blockers,
            require_gate=args.require_gate,
            include_ready=args.include_ready,
        )
        print("\n🛠️ Rewrite Board\n")
        for paper in papers:
            print(
                f"- {paper['name']} | priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                f"rewrite_gain={paper.get('rewrite_priority_gain_total')} | best_round={paper.get('rewrite_best_round_priority_delta')} | "
                f"top_section={paper.get('rewrite_top_section')} | blockers={paper.get('blocker_count')} | "
                f"todo={paper.get('experiment_todo_count')} p0={paper.get('experiment_todo_p0_count')} closure={paper.get('experiment_todo_closure_rate')} | "
                f"repair_ready={paper.get('review_repair_ready_coverage')} binding={paper.get('review_active_binding_coverage')}"
            )
            print(f"  next: {paper.get('suggested_next_step')}")
            print()

        if args.export:
            export_path = manager.export_rewrite_board_markdown(papers, args.export)
            print(f"📝 已导出 rewrite board: {export_path}")
    elif args.command == "repair-board":
        rows = manager.repair_board(
            top_n=args.top,
            target_venue=args.venue,
            priority_tier=args.priority_tier,
            only_ready=args.only_ready,
        )
        print("\n🧩 Repair Board\n")
        for row in rows:
            print(
                f"- {row.get('name')} | {row.get('priority_tier')} {row.get('status')} | "
                f"target={row.get('primary_target_type')}:{row.get('primary_target_id')} | "
                f"role={row.get('role')} | issue={row.get('issue_text')}"
            )
            print(
                f"  actions: {' | '.join(row.get('repair_actions') or []) or 'none'}"
            )
            print(
                f"  verification: {' | '.join(row.get('verification_checks') or []) or 'none'}"
            )
            print()

        if args.export:
            export_path = manager.export_repair_board_markdown(rows, args.export)
            print(f"📝 已导出 repair board: {export_path}")
    elif args.command == "evolution-board":
        rows = manager.evolution_board(
            top_n=args.top,
            status=args.status,
        )
        print("\n🧠 Self-Evolution Board\n")
        for row in rows:
            print(
                f"- {row.get('name')} | status={row.get('status')} | score={row.get('score')} | "
                f"lane={row.get('dominant_lane')} | role={row.get('dominant_role')} | "
                f"lessons={row.get('lesson_count')}"
            )
            if row.get("stage_risks"):
                print(f"  stage_risks={row.get('stage_risks')[:3]}")
            if row.get("top_lessons"):
                print(
                    "  lessons="
                    + " | ".join(
                        str(item.get("recommended_action") or "")
                        for item in row.get("top_lessons") or []
                        if str(item.get("recommended_action") or "").strip()
                    )
                )
            if row.get("next_cycle_defaults"):
                print(f"  next_cycle_defaults={row.get('next_cycle_defaults')}")
            print()
    elif args.command == "submission-dossier":
        result = manager.export_submission_dossier(args.folder, args.output_dir)
        if result["status"] != "success":
            print(f"❌ {result['reason']}")
            sys.exit(1)
        print(f"📦 投稿材料已导出到: {result['output_dir']}")
        print(f"📋 Manifest: {result['manifest']}")
    elif args.command == "readiness-benchmark":
        benchmark = manager.readiness_benchmark(
            target_venue=args.venue,
            max_entries=args.max_entries,
            top_n=args.top,
            include_other_venues=args.include_other_venues,
        )
        summary = benchmark.get("summary", {})
        print("\n🏁 Readiness Benchmark\n")
        print(f"Target venue: {benchmark.get('target_venue')}")
        print(f"Research dir: {benchmark.get('research_root')}")
        print(f"Entries: {summary.get('entries')}")
        print(f"Venue match: {summary.get('venue_match_count')}")
        print(f"Ready: {summary.get('ready_count')}")
        print(f"Gate passed: {summary.get('gate_pass_count')}")
        print(f"Avg benchmark score: {summary.get('avg_benchmark_score')}")
        print(f"Avg submission priority: {summary.get('avg_submission_priority')}")
        print(f"Avg blocker count: {summary.get('avg_blocker_count')}")
        print(
            "Avg process alignment score: "
            f"{summary.get('avg_process_alignment_score')}"
        )
        print(
            "Avg blocked process alignment count: "
            f"{summary.get('avg_process_alignment_blocked_count')}"
        )
        print(f"Avg self-evolution score: {summary.get('avg_self_evolution_score')}")
        print(
            "Avg self-evolution required failures: "
            f"{summary.get('avg_self_evolution_required_failure_count')}"
        )

        top_gaps = summary.get("top_gap_dimensions") or {}
        if top_gaps:
            print("\nTop gap dimensions:")
            for name, count in top_gaps.items():
                print(f"  - {name}: {count}")

        top_categories = summary.get("top_blocker_categories") or {}
        if top_categories:
            print("\nTop blocker categories:")
            for name, count in top_categories.items():
                print(f"  - {name}: {count}")

        top_process_risks = summary.get("top_process_alignment_risks") or {}
        if top_process_risks:
            print("\nTop process alignment risks:")
            for name, count in top_process_risks.items():
                print(f"  - {name}: {count}")

        top_evolution_risks = summary.get("top_self_evolution_risks") or {}
        if top_evolution_risks:
            print("\nTop self-evolution risks:")
            for name, count in top_evolution_risks.items():
                print(f"  - {name}: {count}")

        ranked = benchmark.get("ranked_papers") or []
        if ranked:
            print("\nTop papers:")
            for paper in ranked:
                print(
                    f"- {paper.get('name')} | benchmark={paper.get('benchmark_score')} | "
                    f"status={paper.get('submission_status')} | gate={paper.get('quality_gate_passed')} | "
                    f"priority={paper.get('submission_priority_score')} ({paper.get('submission_priority_tier')}) | "
                    f"blockers={paper.get('blocker_count')} | "
                    f"process_alignment={paper.get('process_alignment_overall_score')} "
                    f"blocked={paper.get('process_alignment_blocked_process_count')} | "
                    f"self_evolution={paper.get('self_evolution_status')}:{paper.get('self_evolution_score')} "
                    f"required_failures={paper.get('self_evolution_required_failure_count')} | "
                    f"venue={paper.get('paper_target_venue')} | "
                    f"match={paper.get('venue_match')}"
                )
                failing_metrics = paper.get("failing_metrics") or []
                if failing_metrics:
                    print(
                        "  gaps: "
                        + ", ".join(
                            f"{item.get('name')}({item.get('gap')})"
                            for item in failing_metrics[:3]
                        )
                    )
                if paper.get("top_blockers"):
                    print("  blockers: " + " | ".join(paper["top_blockers"]))
                if paper.get("recommendation"):
                    print(f"  next: {paper.get('recommendation')}")
                print()

        if args.export:
            export_path = manager.export_readiness_benchmark(benchmark, args.export)
            print(f"📝 已导出 readiness benchmark: {export_path}")
    elif args.command == "pipeline-status":
        rows = manager.pipeline_status(top_n=args.top)
        print("\n🧩 Pipeline Status\n")
        for row in rows:
            print(
                f"- {row['project']} | ready={row.get('ready_count')}/{row.get('artifact_total')} | "
                f"stage_score={row.get('stage_overall_score')} | "
                f"process_alignment={row.get('process_alignment_overall_score')} blocked_processes={row.get('process_alignment_blocked_process_count')} | "
                f"evolution={row.get('self_evolution_status')}:{row.get('self_evolution_score')} "
                f"review_resolution={row.get('review_resolution_rate')} review_binding={row.get('review_target_binding_coverage')} "
                f"repair_ready={row.get('review_repair_ready_coverage')} "
                f"persistent_review_issues={row.get('review_persistent_issue_count')} | "
                f"fallbacks={row.get('fallback_count')} strict={row.get('strict_fallback_count')} | "
                f"template={row.get('template_profile')} ({row.get('template_capability')})"
            )
            print(
                f"  blocked={row.get('blocked_artifacts')} failed={row.get('failed_artifacts')} missing={row.get('missing_artifacts')}"
            )
            print(
                f"  standard_blocked={row.get('blocked_standard_stages')} attention={row.get('attention_standard_stages')} missing={row.get('missing_standard_stages')}"
            )
            if row.get("top_standard_risks"):
                print(f"  top_risks={row.get('top_standard_risks')}")
            if row.get("process_alignment_top_risks"):
                print(f"  process_risks={row.get('process_alignment_top_risks')}")
            if row.get("warnings"):
                print(f"  warnings={row.get('warnings')[:3]}")
            print()
    elif args.command == "stage-standards":
        rows = manager.stage_standards_board(
            top_n=args.top,
            stage=args.stage,
            status=args.status,
        )
        print("\n📏 Stage Standards\n")
        for row in rows:
            print(
                f"- {row.get('project')} | stage={row.get('stage')} | status={row.get('status')} | "
                f"score={row.get('score')} | passed={row.get('passed_criteria_count')}/{row.get('criteria_count')}"
            )
            if row.get("required_failures"):
                print(f"  required_failures={row.get('required_failures')}")
            if row.get("missing_reason"):
                print(f"  missing_reason={row.get('missing_reason')}")
            if row.get("signals"):
                print(f"  signals={row.get('signals')}")
            print()
    elif args.command == "process-board":
        rows = manager.process_board(
            top_n=args.top,
            process=args.process,
            status=args.status,
        )
        print("\n🧭 Process Alignment Board\n")
        for row in rows:
            print(
                f"- {row.get('project')} | process={row.get('process')} | status={row.get('status')} | "
                f"score={row.get('score')} | passed={row.get('passed_criteria_count')}/{row.get('criteria_count')}"
            )
            if row.get("references"):
                print(f"  refs={row.get('references')}")
            if row.get("required_failures"):
                print(f"  required_failures={row.get('required_failures')}")
            if row.get("risks"):
                print(f"  risks={row.get('risks')}")
            if row.get("missing_reason"):
                print(f"  missing_reason={row.get('missing_reason')}")
            if row.get("signals"):
                print(f"  signals={row.get('signals')}")
            print()
    elif args.command == "fallback-board":
        rows = manager.fallback_board(top_n=args.top, stage=args.stage)
        print("\n🧯 Fallback Board\n")
        for row in rows:
            print(
                f"- {row.get('project')} | workflow={row.get('workflow_mode')} | "
                f"fallbacks={row.get('fallback_count')} strict={row.get('strict_fallback_count')}"
            )
            print(
                f"  stages={row.get('stage_counts')} kinds={row.get('kind_counts')}"
            )
            print(
                f"  latest={row.get('latest_stage')} / {row.get('latest_kind')} | reason={row.get('latest_reason')}"
            )
            print(
                f"  metadata={row.get('latest_metadata')} recorded_at={row.get('latest_recorded_at')}"
            )
            print()
    elif args.command == "idea-board":
        rows = manager.idea_board(top_n=args.top, status=args.status)
        print("\n💡 Idea Board\n")
        for row in rows:
            print(
                f"- {row.get('project')}::{row.get('idea_id')} | status={row.get('status')} | "
                f"venue={row.get('target_venue')} | risk={row.get('compute_risk')}"
            )
            print(
                f"  datasets={row.get('datasets')} metrics={row.get('metrics')} baselines={row.get('baselines')}"
            )
            print(f"  mve={row.get('minimum_viable_experiment')}")
            print()
    elif args.command == "experiment-board":
        rows = manager.experiment_board(top_n=args.top, status=args.status)
        print("\n🧪 Experiment Board\n")
        for row in rows:
            summary = row.get("result_summary") or {}
            print(
                f"- {row.get('project')}::{row.get('task_id')} | status={row.get('status')} | "
                f"dataset={row.get('dataset')} | metric={row.get('metric')} | storyline={row.get('entered_storyline')}"
            )
            if summary:
                print(
                    f"  best_metric={summary.get('metric_name')} mean={summary.get('metric_mean')} warnings={summary.get('warnings')}"
                )
            if row.get("error_type") or row.get("error_message"):
                print(f"  error={row.get('error_type')}: {row.get('error_message')}")
            print()
    elif args.command == "figure-board":
        rows = manager.figure_board(
            top_n=args.top,
            include_blocked=not args.ready_only,
        )
        print("\n📈 Figure Board\n")
        for row in rows:
            print(
                f"- {row.get('project')}::{row.get('figure_id')} | status={row.get('status')} | "
                f"claim={row.get('claim_id')} | type={row.get('figure_type')} | slot={row.get('paper_slot')}"
            )
            print(
                f"  data_files={row.get('data_files')} source_records={row.get('source_records')} blocking={row.get('blocking_reasons')}"
            )
            print()
    elif args.command == "source-board":
        rows = manager.source_board(top_n=args.top, archetype=args.archetype)
        print("\n🧭 Source Board\n")
        for row in rows:
            print(
                f"- {row.get('source_name') or row.get('source_key')} | type={row.get('source_type')} | "
                f"archetype={row.get('source_archetype')} | profile={row.get('source_batch_profile')} | "
                f"workflow={row.get('source_workflow_mode')}"
            )
            print(
                f"  runs={row.get('run_count')} ready={row.get('ready_count')} gate={row.get('gate_pass_count')} "
                f"avg_quality={row.get('avg_quality_score')} avg_priority={row.get('avg_submission_priority')} "
                f"avg_fallback={row.get('avg_fallback_count')} strict_fallback={row.get('avg_strict_fallback_count')} "
                f"venue={row.get('target_venue')}"
            )
            print(
                f"  source={row.get('source_value')} latest={row.get('latest_project')} "
                f"fallback_free_rate={row.get('fallback_free_rate')} updated={row.get('updated_at')}"
            )
            print()
    elif args.command == "source-mix":
        advisory = manager.source_mix_advisory(
            desired_policy=args.desired_policy,
            top_n=args.top,
        )
        summary = advisory.get("summary") or {}
        print("\n🧭 Source Mix\n")
        print(f"Desired policy: {advisory.get('desired_policy') or 'n/a'}")
        print(f"Source count: {summary.get('source_count')}")
        print(f"Dominant archetype: {summary.get('dominant_archetype')}")
        print(f"Dominant workflow: {summary.get('dominant_workflow_mode')}")
        print(f"Archetype counts: {summary.get('archetype_counts')}")
        print(f"Workflow counts: {summary.get('workflow_mode_counts')}")
        print(f"Batch profile counts: {summary.get('batch_profile_counts')}")
        if advisory.get("top_sources"):
            print("\nTop sources:")
            for row in advisory.get("top_sources") or []:
                print(
                    f"- {row.get('source_name') or row.get('source_key')} | archetype={row.get('source_archetype')} | "
                    f"workflow={row.get('source_workflow_mode')} | ready={row.get('ready_count')} | "
                    f"gate={row.get('gate_pass_count')} | avg_priority={row.get('avg_submission_priority')}"
                )
        if advisory.get("recommendations"):
            print("\nRecommendations:")
            for item in advisory.get("recommendations") or []:
                print(f"- [{item.get('tier')}] {item.get('recommendation')}")
    elif args.command == "source-next-batch":
        advisory = manager.source_next_batch_advisory(
            desired_policy=args.desired_policy,
            top_n=args.top,
            max_slots=args.max_slots,
        )
        summary = advisory.get("summary") or {}
        cadence = advisory.get("cadence") or {}
        print("\n🧭 Next Batch Source Mix\n")
        print(f"Desired policy: {advisory.get('desired_policy') or 'n/a'}")
        print(f"Source count: {summary.get('source_count')}")
        print(f"Dominant archetype: {summary.get('dominant_archetype')}")
        print(f"Dominant workflow: {summary.get('dominant_workflow_mode')}")
        print(f"Cadence: {cadence.get('label')} | {cadence.get('reason')}")
        if advisory.get("slots"):
            print("\nSlots:")
            for slot in advisory.get("slots") or []:
                print(
                    f"- {slot.get('lane')} | {slot.get('source')} | archetype={slot.get('source_archetype')} | "
                    f"workflow={slot.get('source_workflow_mode')} | profile={slot.get('source_batch_profile')} | "
                    f"share={slot.get('share')}"
                )
                print(
                    f"  ready={slot.get('ready_count')} gate={slot.get('gate_pass_count')} "
                    f"avg_priority={slot.get('avg_submission_priority')} score={slot.get('batch_score')}"
                )
                print(f"  focus={slot.get('focus')}")
                print(f"  rationale={slot.get('rationale')}")
                print()
        if advisory.get("recommendations"):
            print("Recommendations:")
            for item in advisory.get("recommendations") or []:
                print(f"- [{item.get('tier')}] {item.get('recommendation')}")
    elif args.command == "benchmark-trends":
        trends = manager.benchmark_trends(
            target_venue=args.venue,
            max_entries=args.max_entries,
        )
        print("\n📊 Benchmark Trends\n")
        print(f"Target venue: {trends.get('target_venue')}")
        print(f"Summary: {trends.get('summary')}")
        for row in trends.get("timeline") or []:
            print(
                f"- {row.get('date')} | count={row.get('count')} | ready={row.get('ready_count')} | "
                f"gate={row.get('gate_pass_count')} | avg_benchmark={row.get('avg_benchmark_score')} | "
                f"avg_priority={row.get('avg_submission_priority')}"
            )


if __name__ == "__main__":
    main()
