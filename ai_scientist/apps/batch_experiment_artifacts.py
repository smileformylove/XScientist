"""Experiment TODO, ledger, and agenda artifacts for batch workflows."""

from __future__ import annotations

from collections import Counter
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def _safe_read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def _normalize_priority(value: Any, default: str = "P1") -> str:
    text = str(value or default).strip().upper()
    if text in {"P0", "P1", "P2", "P3"}:
        return text
    return default


def _priority_rank(priority: Any) -> int:
    normalized = _normalize_priority(priority)
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(normalized, 3)


def _slugify_token(value: Any) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", str(value or "").strip().lower()).strip("-")
    return token or "paper"


def _extract_self_review_gate_context(paper: Dict[str, Any]) -> dict[str, Any]:
    paper_dir = str(paper.get("paper_dir") or "").strip()
    if not paper_dir:
        return {
            "reasons": [],
            "next_focus": [],
            "ready": None,
            "score": None,
            "unresolved_critical": None,
        }

    root = Path(paper_dir)
    if not root.exists():
        return {
            "reasons": [],
            "next_focus": [],
            "ready": None,
            "score": None,
            "unresolved_critical": None,
        }

    summary = _safe_read_json_dict(root / "self_review_iteration_summary.json")
    improvement_record = _safe_read_json_dict(root / "improvement_record.json")
    final_progress_payload = _safe_read_json_dict(
        root / "self_review_final_progress.json"
    )

    rounds = summary.get("rounds") if isinstance(summary.get("rounds"), list) else []
    latest_round = rounds[-1] if rounds else {}

    latest_gate = summary.get("latest_round_gate")
    if not isinstance(latest_gate, dict):
        latest_gate = (
            dict(latest_round.get("round_gate") or {})
            if isinstance(latest_round, dict)
            else {}
        )
    if not latest_gate and isinstance(improvement_record.get("final_round_gate"), dict):
        latest_gate = dict(improvement_record.get("final_round_gate") or {})

    gate_reasons = _coerce_str_list(latest_gate.get("reasons"))
    next_focus = _coerce_str_list(
        latest_gate.get("next_focus_summaries")
        or latest_gate.get("next_focus_issue_ids")
        or (
            latest_round.get("next_focus_summaries")
            if isinstance(latest_round, dict)
            else []
        )
    )

    gate_score = latest_gate.get("score")
    try:
        gate_score = float(gate_score) if gate_score is not None else None
    except (TypeError, ValueError):
        gate_score = None

    ready = latest_gate.get("ready")
    ready = ready if isinstance(ready, bool) else None

    unresolved_critical = None
    final_progress = final_progress_payload.get("final_progress")
    if isinstance(final_progress, dict) and isinstance(
        final_progress.get("unresolved_critical_count"), int
    ):
        unresolved_critical = int(final_progress.get("unresolved_critical_count"))
    else:
        metrics = latest_gate.get("metrics")
        if isinstance(metrics, dict):
            try:
                unresolved_critical = int(metrics.get("unresolved_critical_count"))
            except (TypeError, ValueError):
                unresolved_critical = None

    return {
        "reasons": gate_reasons[:8],
        "next_focus": next_focus[:6],
        "ready": ready,
        "score": gate_score,
        "unresolved_critical": unresolved_critical,
    }


def _build_paper_experiment_todo_tasks(
    paper: Dict[str, Any], max_tasks: int = 6
) -> list[dict[str, Any]]:
    paper_name = str(
        paper.get("idea_name") or paper.get("paper_type") or "paper"
    ).strip()
    paper_dir = str(paper.get("paper_dir") or "").strip()
    gate = _extract_self_review_gate_context(paper)
    gate_reasons = gate.get("reasons") or []
    next_focus = gate.get("next_focus") or []

    raw_tasks: list[dict[str, Any]] = []
    gate_reason_templates: dict[str, dict[str, str]] = {
        "critical_issues_unresolved": {
            "priority": "P0",
            "focus": "soundness",
            "action": "Run targeted validation experiments that directly close each unresolved critical issue.",
            "success_criterion": "Self-review unresolved critical count becomes 0 in the next round gate.",
            "reason": "round gate indicates unresolved critical issues",
            "source": "self_review_round_gate",
            "source_signal": "critical_issues_unresolved",
            "completion_rule": "gate_reason_cleared:critical_issues_unresolved",
        },
        "high_value_coverage_low": {
            "priority": "P0",
            "focus": "high_value_coverage",
            "action": "Prioritize experiments for unresolved P0/P1 issues and explicitly map outputs to addressed issue ids.",
            "success_criterion": "High-value coverage ratio reaches at least 0.80 in the next rewrite pass.",
            "reason": "high-value issue coverage is too low",
            "source": "self_review_round_gate",
            "source_signal": "high_value_coverage_low",
            "completion_rule": "gate_reason_cleared:high_value_coverage_low",
        },
        "rewrite_coverage_low": {
            "priority": "P1",
            "focus": "rewrite_trace",
            "action": "Execute an issue-linked rewrite pass where each change is traceable to recommended targets.",
            "success_criterion": "Round gate no longer reports low rewrite coverage.",
            "reason": "issue-linked rewrite coverage remains low",
            "source": "self_review_round_gate",
            "source_signal": "rewrite_coverage_low",
            "completion_rule": "gate_reason_cleared:rewrite_coverage_low",
        },
        "persistent_issues_high": {
            "priority": "P1",
            "focus": "persistent_issues",
            "action": "Design section-specific experiments or analyses to break persistent issue loops.",
            "success_criterion": "Persistent issue count decreases in the next self-review round.",
            "reason": "persistent issues remain high across rounds",
            "source": "self_review_round_gate",
            "source_signal": "persistent_issues_high",
            "completion_rule": "gate_reason_cleared:persistent_issues_high",
        },
        "latex_compile_failed": {
            "priority": "P1",
            "focus": "pipeline_stability",
            "action": "Stabilize LaTeX/figure pipeline and rerun evidence generation before further rewrites.",
            "success_criterion": "Round gate no longer reports latex compile failures.",
            "reason": "build instability blocks reliable iteration",
            "source": "self_review_round_gate",
            "source_signal": "latex_compile_failed",
            "completion_rule": "gate_reason_cleared:latex_compile_failed",
        },
    }

    for reason in gate_reasons:
        template = gate_reason_templates.get(reason)
        if template:
            raw_tasks.append(dict(template))
            continue
        if reason.startswith("round<"):
            min_rounds = (
                re.sub(r"[^0-9]", "", reason) if isinstance(reason, str) else ""
            )
            raw_tasks.append(
                {
                    "priority": "P1",
                    "focus": "round_budget",
                    "action": "Extend one additional focused improvement round on unresolved high-value evidence gaps.",
                    "success_criterion": "At least one top unresolved issue is fully closed in the added round.",
                    "reason": f"round budget not yet sufficient ({reason})",
                    "source": "self_review_round_gate",
                    "source_signal": reason,
                    "completion_rule": (
                        f"round_index_ge:{min_rounds}"
                        if min_rounds
                        else "round_gate_ready"
                    ),
                }
            )

    for focus_item in next_focus[:3]:
        raw_tasks.append(
            {
                "priority": "P1",
                "focus": "self_review_focus",
                "action": f"Close next-focus self-review item: {focus_item}",
                "success_criterion": "This focus item no longer appears in next_focus_summaries.",
                "reason": "self-review gate surfaced this as top unresolved focus",
                "source": "self_review_next_focus",
                "source_signal": focus_item,
                "completion_rule": f"next_focus_cleared:{focus_item}",
            }
        )

    revision_actions = paper.get("revision_actions") or []
    for item in revision_actions:
        if not isinstance(item, dict):
            continue
        action_text = str(item.get("action") or "").strip()
        if not action_text:
            continue
        focus = str(item.get("focus") or "experiments").strip()
        focus_lower = focus.lower()
        priority = _normalize_priority(item.get("priority"), default="P1")
        if (
            focus_lower
            not in {
                "experiments",
                "rigor",
                "results",
                "analysis",
                "evidence",
                "claim_support",
                "claims",
            }
            and _priority_rank(priority) > 1
        ):
            continue
        reason_text = str(item.get("reason") or "").strip()
        success_criterion = (
            "Add at least one new or stronger quantitative result and cite it in Results and Abstract."
            if focus_lower in {"experiments", "rigor", "results", "analysis"}
            else "Revision action is resolved and no longer appears in top-priority action list."
        )
        raw_tasks.append(
            {
                "priority": priority,
                "focus": focus,
                "action": action_text,
                "success_criterion": success_criterion,
                "reason": reason_text
                or "quality revision action indicates unresolved evidence or rigor gap",
                "source": "revision_actions",
                "source_signal": action_text,
                "completion_rule": "round_gate_ready",
            }
        )

    unsupported_claims = (
        int(paper.get("unsupported_claims_count"))
        if isinstance(paper.get("unsupported_claims_count"), int)
        else None
    )
    if unsupported_claims and unsupported_claims > 0:
        raw_tasks.append(
            {
                "priority": "P0" if unsupported_claims >= 3 else "P1",
                "focus": "claim_support",
                "action": "Run evidence-strengthening analyses for unsupported claims and bind each key claim to explicit figures/tables.",
                "success_criterion": "Unsupported claims count decreases to 0 or strictly below current level.",
                "reason": f"unsupported claim count remains high ({unsupported_claims})",
                "source": "evidence_metrics",
                "source_signal": "unsupported_claims_count",
                "completion_rule": "unresolved_critical_zero",
            }
        )

    evidence_density = paper.get("evidence_density_score")
    if isinstance(evidence_density, (int, float)) and float(evidence_density) < 2.0:
        raw_tasks.append(
            {
                "priority": "P1",
                "focus": "evidence_density",
                "action": "Add high-signal quantitative evidence blocks that directly support the lead contribution.",
                "success_criterion": "Evidence density score rises to at least 2.0 in the next quality pass.",
                "reason": f"evidence density is below target ({float(evidence_density):.2f} < 2.00)",
                "source": "evidence_metrics",
                "source_signal": "evidence_density_score",
                "completion_rule": "high_value_coverage_ge:0.8",
            }
        )

    deduped: dict[str, dict[str, Any]] = {}
    for item in raw_tasks:
        action_key = re.sub(
            r"\s+",
            " ",
            str(item.get("action") or "").strip().lower(),
        )
        if not action_key:
            continue
        existing = deduped.get(action_key)
        if not existing or _priority_rank(item.get("priority")) < _priority_rank(
            existing.get("priority")
        ):
            deduped[action_key] = item

    source_rank = {
        "self_review_round_gate": 0,
        "self_review_next_focus": 1,
        "revision_actions": 2,
        "evidence_metrics": 3,
    }
    ordered = sorted(
        deduped.values(),
        key=lambda task: (
            _priority_rank(task.get("priority")),
            source_rank.get(str(task.get("source")), 9),
            str(task.get("focus") or ""),
            str(task.get("action") or ""),
        ),
    )

    prefix = (
        f"idea{paper.get('idea_idx')}"
        if paper.get("idea_idx") is not None
        else _slugify_token(paper_name)
    )
    tasks: list[dict[str, Any]] = []
    for idx, task in enumerate(ordered[: max(1, int(max_tasks))], start=1):
        tasks.append(
            {
                "task_id": f"{prefix}-T{idx:02d}",
                "paper": paper_name,
                "paper_dir": paper_dir or None,
                "priority": _normalize_priority(task.get("priority"), default="P1"),
                "focus": str(task.get("focus") or "experiments"),
                "action": str(task.get("action") or "").strip(),
                "success_criterion": str(task.get("success_criterion") or "").strip(),
                "reason": str(task.get("reason") or "").strip(),
                "source": str(task.get("source") or "derived"),
                "source_signal": str(task.get("source_signal") or "").strip(),
                "completion_rule": str(task.get("completion_rule") or "").strip(),
            }
        )
    return tasks


def _build_batch_experiment_todo(
    report: Dict[str, Any],
    *,
    max_tasks_per_paper: int = 6,
    max_papers: int = 8,
) -> Dict[str, Any]:
    top_papers = report.get("quality_summary", {}).get("top_papers")
    completed = report.get("completed_papers") or []
    candidates = (
        top_papers if isinstance(top_papers, list) and top_papers else completed
    )

    if not isinstance(candidates, list):
        candidates = []

    tasks: list[dict[str, Any]] = []
    paper_summaries: list[dict[str, Any]] = []
    for paper in candidates[: max(1, int(max_papers))]:
        if not isinstance(paper, dict):
            continue
        paper_tasks = _build_paper_experiment_todo_tasks(
            paper, max_tasks=max_tasks_per_paper
        )
        if not paper_tasks:
            continue
        p0_count = sum(task.get("priority") == "P0" for task in paper_tasks)
        p1_count = sum(task.get("priority") == "P1" for task in paper_tasks)
        paper_summaries.append(
            {
                "paper": paper.get("idea_name") or paper.get("paper_type"),
                "paper_dir": paper.get("paper_dir"),
                "task_count": len(paper_tasks),
                "p0_count": p0_count,
                "p1_count": p1_count,
                "top_task": paper_tasks[0].get("action") if paper_tasks else "",
            }
        )
        tasks.extend(paper_tasks)

    tasks.sort(
        key=lambda task: (
            _priority_rank(task.get("priority")),
            str(task.get("paper") or ""),
            str(task.get("task_id") or ""),
        )
    )
    counts = {
        "total_tasks": len(tasks),
        "p0_tasks": sum(task.get("priority") == "P0" for task in tasks),
        "p1_tasks": sum(task.get("priority") == "P1" for task in tasks),
        "p2_tasks": sum(task.get("priority") == "P2" for task in tasks),
        "p3_tasks": sum(task.get("priority") == "P3" for task in tasks),
        "papers_with_tasks": len(paper_summaries),
    }
    return {
        "generated_at": datetime.now().isoformat(),
        "counts": counts,
        "paper_summaries": paper_summaries,
        "tasks": tasks,
    }


def _build_batch_experiment_todo_markdown(todo: Dict[str, Any]) -> str:
    counts = todo.get("counts") or {}
    lines = [
        "# Batch Experiment TODO",
        "",
        f"- Generated at: {todo.get('generated_at')}",
        f"- Total tasks: {counts.get('total_tasks', 0)}",
        f"- P0 tasks: {counts.get('p0_tasks', 0)}",
        f"- P1 tasks: {counts.get('p1_tasks', 0)}",
        f"- Papers with tasks: {counts.get('papers_with_tasks', 0)}",
        "",
        "## Paper Backlog",
    ]
    paper_summaries = todo.get("paper_summaries") or []
    if paper_summaries:
        for item in paper_summaries:
            lines.append(
                f"- {item.get('paper')}: total={item.get('task_count')} p0={item.get('p0_count')} p1={item.get('p1_count')} top={item.get('top_task')}"
            )
    else:
        lines.append("- No executable experiment tasks extracted.")

    lines.extend(["", "## Executable Tasks"])
    tasks = todo.get("tasks") or []
    if tasks:
        for task in tasks:
            completion_rule = str(task.get("completion_rule") or "").strip()
            completion_part = f" | rule={completion_rule}" if completion_rule else ""
            lines.append(
                f"- [{task.get('priority')}] {task.get('task_id')} {task.get('paper')}: {task.get('action')} | success={task.get('success_criterion')} | reason={task.get('reason')} | source={task.get('source')}{completion_part}"
            )
    else:
        lines.append("- No tasks.")
    return "\n".join(lines) + "\n"


def _write_per_paper_experiment_todo_artifacts(todo: Dict[str, Any]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for task in todo.get("tasks") or []:
        paper_dir = str(task.get("paper_dir") or "").strip()
        if not paper_dir:
            continue
        grouped.setdefault(paper_dir, []).append(task)

    generated_at = todo.get("generated_at")
    for paper_dir, tasks in grouped.items():
        root = Path(paper_dir)
        if not root.exists():
            continue
        payload = {
            "generated_at": generated_at,
            "counts": {
                "total_tasks": len(tasks),
                "p0_tasks": sum(task.get("priority") == "P0" for task in tasks),
                "p1_tasks": sum(task.get("priority") == "P1" for task in tasks),
            },
            "tasks": tasks,
        }
        (root / "experiment_todo.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        lines = [
            "# Experiment TODO",
            "",
            f"- Generated at: {generated_at}",
            f"- Total tasks: {payload['counts']['total_tasks']}",
            f"- P0 tasks: {payload['counts']['p0_tasks']}",
            f"- P1 tasks: {payload['counts']['p1_tasks']}",
            "",
            "## Tasks",
        ]
        for task in tasks:
            completion_rule = str(task.get("completion_rule") or "").strip()
            completion_part = f" | rule={completion_rule}" if completion_rule else ""
            lines.append(
                f"- [{task.get('priority')}] {task.get('task_id')}: {task.get('action')} | success={task.get('success_criterion')} | reason={task.get('reason')} | source={task.get('source')}{completion_part}"
            )
        (root / "experiment_todo.md").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )


def _annotate_report_with_experiment_todo(
    report: Dict[str, Any], todo: Dict[str, Any]
) -> None:
    paper_map: dict[str, dict[str, Any]] = {}
    for item in todo.get("paper_summaries") or []:
        key = str(item.get("paper_dir") or "").strip()
        if not key:
            continue
        paper_map[key] = item

    for bucket in ("completed_papers", "failed_papers"):
        papers = report.get(bucket) or []
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            key = str(paper.get("paper_dir") or "").strip()
            if not key:
                continue
            stats = paper_map.get(key) or {}
            paper["experiment_todo_count"] = int(stats.get("task_count") or 0)
            paper["experiment_todo_p0_count"] = int(stats.get("p0_count") or 0)
            paper["experiment_todo_top_action"] = str(stats.get("top_task") or "")
            paper["experiment_todo_file"] = str(Path(key) / "experiment_todo.json")


def _classify_batch_experiment_outcome(paper: Dict[str, Any]) -> dict[str, Any]:
    if paper.get("status") != "success":
        return {
            "decision": "crash",
            "reasons": [
                paper.get("stage") or paper.get("error") or "paper generation failed"
            ],
        }

    reasons: list[str] = []
    priority = paper.get("submission_priority_score")
    blockers = paper.get("blocker_count")
    gate_passed = paper.get("quality_gate_passed")
    unsupported_claims = paper.get("unsupported_claims_count")
    evidence_density = paper.get("evidence_density_score")

    if paper.get("submission_acceptance_passed") is True:
        reasons.append("submission acceptance bar already passed")
    if gate_passed is True:
        reasons.append("quality gate passed")
    if isinstance(priority, (int, float)) and priority >= 85:
        reasons.append("submission priority is already high")
    if isinstance(blockers, int) and blockers <= 1:
        reasons.append("blocker count is low")
    if isinstance(evidence_density, (int, float)) and evidence_density >= 2.0:
        reasons.append("evidence density is acceptable")

    if paper.get("submission_acceptance_passed") is True or (
        gate_passed is True
        and isinstance(priority, (int, float))
        and priority >= 80
        and isinstance(blockers, int)
        and blockers <= 2
    ):
        return {
            "decision": "keep",
            "reasons": reasons
            or [
                "quality indicators are strong enough to keep iterating from this result"
            ],
        }

    discard_reasons = []
    if gate_passed is False:
        discard_reasons.append("quality gate failed")
    if isinstance(blockers, int) and blockers >= 5:
        discard_reasons.append("too many blockers remain")
    if isinstance(unsupported_claims, int) and unsupported_claims >= 3:
        discard_reasons.append("unsupported claim count is too high")
    if discard_reasons:
        return {"decision": "discard", "reasons": discard_reasons}

    return {
        "decision": "keep",
        "reasons": reasons
        or ["result is mixed but still promising enough to keep for further iteration"],
    }


def _build_batch_experiment_ledger_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for paper in report.get("completed_papers", []):
        outcome = _classify_batch_experiment_outcome(paper)
        rows.append(
            {
                "idea_idx": paper.get("idea_idx"),
                "idea_name": paper.get("idea_name"),
                "paper_type": paper.get("paper_type"),
                "target_venue": paper.get("target_venue"),
                "decision": outcome.get("decision"),
                "priority": paper.get("submission_priority_score"),
                "blockers": paper.get("blocker_count"),
                "gate": paper.get("quality_gate_passed"),
                "reason": " | ".join(outcome.get("reasons") or []),
            }
        )
    for paper in report.get("failed_papers", []):
        outcome = _classify_batch_experiment_outcome(paper)
        rows.append(
            {
                "idea_idx": paper.get("idea_idx"),
                "idea_name": paper.get("idea_name"),
                "paper_type": paper.get("paper_type"),
                "target_venue": paper.get("target_venue"),
                "decision": outcome.get("decision"),
                "priority": paper.get("submission_priority_score"),
                "blockers": paper.get("blocker_count"),
                "gate": paper.get("quality_gate_passed"),
                "reason": " | ".join(outcome.get("reasons") or []),
            }
        )
    return rows


def _build_batch_experiment_ledger_tsv(rows: List[Dict[str, Any]]) -> str:
    header = [
        "idea_idx",
        "idea_name",
        "paper_type",
        "target_venue",
        "decision",
        "priority",
        "blockers",
        "gate",
        "reason",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in header))
    return "\n".join(lines) + "\n"


def _build_batch_experiment_agenda(report: Dict[str, Any]) -> Dict[str, Any]:
    rows = _build_batch_experiment_ledger_rows(report)
    counts = Counter(row.get("decision") for row in rows)
    priorities: list[dict[str, str]] = []

    top_papers = report.get("quality_summary", {}).get("top_papers", []) or []
    for paper in top_papers[:5]:
        revision_actions = paper.get("revision_actions") or []
        experiment_actions = [
            item
            for item in revision_actions
            if str(item.get("focus") or "").lower()
            in {"experiments", "rigor", "results", "analysis"}
        ]
        if experiment_actions:
            for item in experiment_actions[:2]:
                priorities.append(
                    {
                        "paper": paper.get("idea_name") or paper.get("paper_type"),
                        "priority": str(item.get("priority") or "P1"),
                        "action": str(item.get("action") or ""),
                        "reason": str(item.get("reason") or ""),
                    }
                )
        elif (
            isinstance(paper.get("unsupported_claims_count"), int)
            and paper.get("unsupported_claims_count") > 0
        ):
            priorities.append(
                {
                    "paper": paper.get("idea_name") or paper.get("paper_type"),
                    "priority": "P1",
                    "action": "Run evidence-strengthening experiments or ablations for the strongest unsupported claims.",
                    "reason": "unsupported claims remain in the current draft",
                }
            )
        elif (
            isinstance(paper.get("evidence_density_score"), (int, float))
            and paper.get("evidence_density_score") < 2.0
        ):
            priorities.append(
                {
                    "paper": paper.get("idea_name") or paper.get("paper_type"),
                    "priority": "P1",
                    "action": "Add one or two higher-signal figures/tables that directly support the lead contribution.",
                    "reason": "evidence density is still thin for a submission-grade story",
                }
            )

    failed_stage_counts = Counter(
        str(item.get("stage") or "unknown") for item in report.get("failed_papers", [])
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "counts": dict(counts),
        "failed_stages": dict(failed_stage_counts),
        "priority_experiments": priorities[:8],
    }


def _build_batch_experiment_agenda_markdown(agenda: Dict[str, Any]) -> str:
    lines = [
        "# Batch Experiment Agenda",
        "",
        f"- Generated at: {agenda.get('generated_at')}",
        f"- Keep: {(agenda.get('counts') or {}).get('keep', 0)}",
        f"- Discard: {(agenda.get('counts') or {}).get('discard', 0)}",
        f"- Crash: {(agenda.get('counts') or {}).get('crash', 0)}",
        "",
        "## Priority Experiments",
    ]
    priorities = agenda.get("priority_experiments") or []
    if priorities:
        for item in priorities:
            lines.append(
                f"- [{item.get('priority')}] {item.get('paper')}: {item.get('action')} ({item.get('reason')})"
            )
    else:
        lines.append("- No new experiment agenda items extracted.")

    lines.extend(["", "## Failure Hotspots"])
    failed_stages = agenda.get("failed_stages") or {}
    if failed_stages:
        for stage, count in sorted(
            failed_stages.items(), key=lambda item: item[1], reverse=True
        ):
            lines.append(f"- {stage}: {count}")
    else:
        lines.append("- No failure hotspots recorded.")
    return "\n".join(lines) + "\n"
