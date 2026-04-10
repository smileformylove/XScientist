from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat()


def _normalize_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().lower())
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\-_ ]+", "", text)
    return text.strip()


def _normalize_priority(value: Any) -> str:
    text = str(value or "P1").strip().upper()
    if text in {"P0", "P1", "P2", "P3"}:
        return text
    return "P1"


def _safe_read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _coerce_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed


def _coerce_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return result


def load_experiment_todo_payload(paper_dir: str | Path) -> dict[str, Any]:
    root = Path(paper_dir)
    payload = _safe_read_json_dict(root / "experiment_todo.json")
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        tasks = []
    normalized_tasks: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            continue
        task_id = str(task.get("task_id") or f"T{idx:02d}").strip()
        action = str(task.get("action") or "").strip()
        if not action:
            continue
        normalized_tasks.append(
            {
                "task_id": task_id,
                "priority": _normalize_priority(task.get("priority")),
                "focus": str(task.get("focus") or "").strip(),
                "action": action,
                "reason": str(task.get("reason") or "").strip(),
                "source": str(task.get("source") or "").strip(),
                "source_signal": str(task.get("source_signal") or "").strip(),
                "completion_rule": str(task.get("completion_rule") or "").strip(),
            }
        )
    payload["tasks"] = normalized_tasks
    payload["file"] = str(root / "experiment_todo.json")
    return payload


def summarize_todo_baseline(todo_payload: dict[str, Any]) -> dict[str, Any]:
    tasks = todo_payload.get("tasks") or []
    p0_total = sum(
        _normalize_priority(item.get("priority")) == "P0"
        for item in tasks
        if isinstance(item, dict)
    )
    return {
        "total_tasks": len(tasks),
        "p0_tasks": int(p0_total),
    }


def bootstrap_todo_tasks_from_round_gate(
    round_gate: dict[str, Any] | None,
    *,
    prefix: str = "bootstrap",
    max_tasks: int = 8,
) -> list[dict[str, Any]]:
    gate = dict(round_gate or {}) if isinstance(round_gate, dict) else {}
    reasons = _coerce_str_list(gate.get("reasons"))
    next_focus = _coerce_str_list(
        gate.get("next_focus_summaries") or gate.get("next_focus_issue_ids")
    )

    tasks: list[dict[str, Any]] = []
    for idx, reason in enumerate(reasons[: max_tasks], start=1):
        priority = "P0" if reason in {"critical_issues_unresolved", "high_value_coverage_low"} else "P1"
        tasks.append(
            {
                "task_id": f"{prefix}-B{idx:02d}",
                "priority": priority,
                "focus": "self_review_gate",
                "action": f"Resolve self-review round-gate blocker: {reason}",
                "reason": "auto-bootstrapped from round gate reasons",
                "source": "self_review_round_gate",
                "source_signal": reason,
                "completion_rule": f"gate_reason_cleared:{reason}",
            }
        )

    room = max(0, max_tasks - len(tasks))
    for idx, focus in enumerate(next_focus[:room], start=1):
        tasks.append(
            {
                "task_id": f"{prefix}-F{idx:02d}",
                "priority": "P1",
                "focus": "self_review_focus",
                "action": f"Close next-focus item: {focus}",
                "reason": "auto-bootstrapped from round-gate focus",
                "source": "self_review_next_focus",
                "source_signal": focus,
                "completion_rule": f"next_focus_cleared:{focus}",
            }
        )
    return tasks


def _extract_gate_signals(
    round_gate: dict[str, Any] | None, issue_progress: dict[str, Any] | None
) -> dict[str, Any]:
    gate = dict(round_gate or {}) if isinstance(round_gate, dict) else {}
    reasons = set(_coerce_str_list(gate.get("reasons")))
    next_focus = _coerce_str_list(
        gate.get("next_focus_summaries") or gate.get("next_focus_issue_ids")
    )
    metrics = gate.get("metrics") if isinstance(gate.get("metrics"), dict) else {}
    unresolved_critical = _coerce_int(metrics.get("unresolved_critical_count"))
    if unresolved_critical is None and isinstance(issue_progress, dict):
        unresolved_critical = _coerce_int(issue_progress.get("unresolved_critical_count"))
    persistent_count = _coerce_int(metrics.get("persistent_issue_count"))
    if persistent_count is None and isinstance(issue_progress, dict):
        persistent_count = _coerce_int(issue_progress.get("persistent_issue_count"))
    coverage_ratio = _coerce_float(metrics.get("coverage_ratio"))
    high_value_coverage = _coerce_float(metrics.get("high_value_coverage_ratio"))
    return {
        "ready": bool(gate.get("ready")) if isinstance(gate.get("ready"), bool) else False,
        "score": _coerce_float(gate.get("score")),
        "reasons": reasons,
        "next_focus": next_focus,
        "next_focus_norm": [_normalize_text(item) for item in next_focus if item],
        "round_index": _coerce_int(gate.get("round_index")),
        "unresolved_critical": unresolved_critical,
        "persistent_issue_count": persistent_count,
        "coverage_ratio": coverage_ratio,
        "high_value_coverage_ratio": high_value_coverage,
    }


def _match_rule(task: dict[str, Any], signals: dict[str, Any]) -> bool:
    rule = str(task.get("completion_rule") or "").strip()
    if not rule:
        return False

    if rule.startswith("gate_reason_cleared:"):
        reason = rule.split(":", 1)[1].strip()
        if reason:
            return reason not in set(signals.get("reasons") or set())
        return False

    if rule.startswith("round_index_ge:"):
        target = _coerce_int(rule.split(":", 1)[1].strip())
        current = _coerce_int(signals.get("round_index"))
        if target is None or current is None:
            return False
        return current >= target

    if rule.startswith("next_focus_cleared:"):
        focus = _normalize_text(rule.split(":", 1)[1])
        if not focus:
            return False
        focus_items = signals.get("next_focus_norm") or []
        return not any(focus in item or item in focus for item in focus_items)

    if rule == "round_gate_ready":
        return bool(signals.get("ready"))

    if rule == "unresolved_critical_zero":
        unresolved = _coerce_int(signals.get("unresolved_critical"))
        return unresolved is not None and unresolved <= 0

    if rule.startswith("high_value_coverage_ge:"):
        target = _coerce_float(rule.split(":", 1)[1].strip())
        value = _coerce_float(signals.get("high_value_coverage_ratio"))
        if target is None or value is None:
            return False
        return value >= target

    if rule.startswith("coverage_ge:"):
        target = _coerce_float(rule.split(":", 1)[1].strip())
        value = _coerce_float(signals.get("coverage_ratio"))
        if target is None or value is None:
            return False
        return value >= target

    return False


def _fallback_match(task: dict[str, Any], signals: dict[str, Any]) -> bool:
    source = str(task.get("source") or "").strip().lower()
    source_signal = str(task.get("source_signal") or "").strip()
    reasons = set(signals.get("reasons") or set())
    next_focus_norm = signals.get("next_focus_norm") or []

    if source == "self_review_round_gate":
        if source_signal:
            return source_signal not in reasons
        return bool(signals.get("ready"))

    if source == "self_review_next_focus":
        if source_signal:
            target = _normalize_text(source_signal)
            return not any(target in item or item in target for item in next_focus_norm)
        return bool(signals.get("ready"))

    if source in {"revision_actions", "evidence_metrics"}:
        return bool(signals.get("ready"))

    return False


def evaluate_todo_progress_snapshot(
    todo_payload: dict[str, Any],
    *,
    round_gate: dict[str, Any] | None = None,
    issue_progress: dict[str, Any] | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    tasks = todo_payload.get("tasks") or []
    signals = _extract_gate_signals(round_gate, issue_progress)
    if round_index is not None:
        signals["round_index"] = int(round_index)

    task_states: list[dict[str, Any]] = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        closed = _match_rule(item, signals) or _fallback_match(item, signals)
        task_states.append(
            {
                "task_id": str(item.get("task_id") or "").strip(),
                "priority": _normalize_priority(item.get("priority")),
                "status": "closed" if closed else "unresolved",
                "action": str(item.get("action") or "").strip(),
                "source": str(item.get("source") or "").strip(),
                "completion_rule": str(item.get("completion_rule") or "").strip(),
            }
        )

    closed = [item for item in task_states if item.get("status") == "closed"]
    unresolved = [item for item in task_states if item.get("status") == "unresolved"]
    p0_states = [item for item in task_states if item.get("priority") == "P0"]
    p0_closed = [item for item in p0_states if item.get("status") == "closed"]
    p0_unresolved = [item for item in p0_states if item.get("status") == "unresolved"]

    closure_rate = round(len(closed) / len(task_states), 4) if task_states else 1.0
    p0_closure_rate = round(len(p0_closed) / len(p0_states), 4) if p0_states else 1.0

    return {
        "generated_at": _now_iso(),
        "round_index": _coerce_int(signals.get("round_index")),
        "gate_ready": bool(signals.get("ready")),
        "gate_score": _coerce_float(signals.get("score")),
        "active_gate_reasons": sorted(list(signals.get("reasons") or set())),
        "active_next_focus": signals.get("next_focus") or [],
        "counts": {
            "total_tasks": len(task_states),
            "closed_tasks": len(closed),
            "unresolved_tasks": len(unresolved),
            "p0_total": len(p0_states),
            "p0_closed": len(p0_closed),
            "p0_unresolved": len(p0_unresolved),
        },
        "closure_rate": closure_rate,
        "p0_closure_rate": p0_closure_rate,
        "closed_task_ids": [item.get("task_id") for item in closed if item.get("task_id")],
        "unresolved_task_ids": [
            item.get("task_id") for item in unresolved if item.get("task_id")
        ],
        "top_unresolved_actions": [
            item.get("action") for item in unresolved[:3] if item.get("action")
        ],
        "task_states": task_states,
    }


def build_todo_progress_payload(
    todo_payload: dict[str, Any],
    *,
    round_snapshots: list[dict[str, Any]] | None = None,
    final_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = summarize_todo_baseline(todo_payload)
    return {
        "generated_at": _now_iso(),
        "baseline_file": todo_payload.get("file"),
        "baseline_counts": baseline,
        "round_snapshots": list(round_snapshots or []),
        "final_snapshot": dict(final_snapshot or {}),
    }


def render_todo_progress_markdown(payload: dict[str, Any]) -> str:
    baseline = payload.get("baseline_counts") or {}
    final_snapshot = payload.get("final_snapshot") or {}
    final_counts = final_snapshot.get("counts") or {}
    lines = [
        "# Experiment TODO Progress",
        "",
        f"- Generated at: {payload.get('generated_at')}",
        f"- Baseline total tasks: {baseline.get('total_tasks', 0)}",
        f"- Baseline P0 tasks: {baseline.get('p0_tasks', 0)}",
        f"- Final closure rate: {final_snapshot.get('closure_rate')}",
        f"- Final P0 closure rate: {final_snapshot.get('p0_closure_rate')}",
        f"- Final unresolved tasks: {final_counts.get('unresolved_tasks', 0)}",
        f"- Final unresolved P0: {final_counts.get('p0_unresolved', 0)}",
        "",
        "## Round Snapshots",
    ]
    rounds = payload.get("round_snapshots") or []
    if rounds:
        for item in rounds:
            counts = item.get("counts") or {}
            lines.append(
                f"- Round {item.get('round_index')}: closure={item.get('closure_rate')} p0_closure={item.get('p0_closure_rate')} unresolved={counts.get('unresolved_tasks', 0)} reasons={item.get('active_gate_reasons')}"
            )
    else:
        lines.append("- No round snapshots.")

    lines.extend(["", "## Final Unresolved Actions"])
    actions = final_snapshot.get("top_unresolved_actions") or []
    if actions:
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- None.")
    return "\n".join(lines) + "\n"


def save_todo_progress_artifacts(
    paper_dir: str | Path,
    payload: dict[str, Any],
) -> dict[str, str]:
    root = Path(paper_dir)
    json_path = root / "experiment_todo_progress.json"
    md_path = root / "experiment_todo_progress.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    md_path.write_text(render_todo_progress_markdown(payload), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
    }
