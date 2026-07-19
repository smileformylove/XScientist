"""Pure data transformation and HTML rendering for the daemon dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _render_section(title: str, content: str, *, open_by_default: bool = True) -> str:
    open_attr = " open" if open_by_default else ""
    return f"<details class='panel'{open_attr}><summary>{_html_escape(title)}</summary><div class='panel-body'>{content}</div></details>"


def _sparkline_svg(
    values: list[float], *, width: int = 280, height: int = 48, color: str = "#7fb3ff"
) -> str:
    if not values:
        return "<p class='small'>No recent data.</p>"
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-9)
    points = []
    for idx, value in enumerate(values):
        x = (idx / max(1, len(values) - 1)) * (width - 8) + 4
        y = height - (((value - min_v) / span) * (height - 12) + 6)
        points.append(f"{x:.1f},{y:.1f}")
    return (
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='trend'>"
        f"<polyline fill='none' stroke='{color}' stroke-width='3' points='{' '.join(points)}' />"
        "</svg>"
    )


def _render_trend_card(title: str, values: list[float], *, latest: Any = None) -> str:
    latest_text = f"<div class='trend-latest'>{_html_escape(latest if latest is not None else (values[-1] if values else 'n/a'))}</div>"
    return (
        "<div class='trend-card'>"
        f"<div class='trend-title'>{_html_escape(title)}</div>"
        f"{latest_text}"
        f"{_sparkline_svg(values)}"
        "</div>"
    )


def _build_recent_trend_metrics(
    cycle_history: list[dict[str, Any]],
) -> dict[str, list[float]]:
    closure_series: list[float] = []
    backlog_series: list[float] = []
    p0_backlog_series: list[float] = []
    gate_ready_series: list[float] = []
    for item in cycle_history:
        feedback = item.get("active_source_feedback") or {}
        closure = feedback.get("avg_experiment_todo_closure_rate")
        backlog = feedback.get("avg_experiment_todo")
        p0_backlog = feedback.get("avg_experiment_todo_p0")
        gate_ready = feedback.get("self_review_gate_ready_rate")
        if isinstance(closure, (int, float)):
            closure_series.append(float(closure))
        if isinstance(backlog, (int, float)):
            backlog_series.append(float(backlog))
        if isinstance(p0_backlog, (int, float)):
            p0_backlog_series.append(float(p0_backlog))
        if isinstance(gate_ready, (int, float)):
            gate_ready_series.append(float(gate_ready))
    return {
        "submission_board_items": [
            float((item.get("views") or {}).get("submission_board_items") or 0)
            for item in cycle_history
        ],
        "rewrite_board_items": [
            float((item.get("views") or {}).get("rewrite_board_items") or 0)
            for item in cycle_history
        ],
        "shortlist_items": [
            float((item.get("views") or {}).get("shortlist_items") or 0)
            for item in cycle_history
        ],
        "duration_seconds": [
            float(item.get("duration_seconds") or 0) for item in cycle_history
        ],
        "returncode": [float(item.get("returncode") or 0) for item in cycle_history],
        "experiment_todo_closure_rate": closure_series,
        "experiment_todo_backlog": backlog_series,
        "experiment_todo_p0_backlog": p0_backlog_series,
        "self_review_gate_ready_rate": gate_ready_series,
    }


def _html_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _path_to_href(value: Any) -> str:
    if not value:
        return ""
    try:
        return Path(str(value)).expanduser().resolve().as_uri()
    except Exception:
        return ""


def _render_stat_cards(cards: list[tuple[str, Any]]) -> str:
    items = []
    for label, value in cards:
        items.append(
            f"<div class='stat-card'><div class='stat-label'>{_html_escape(label)}</div><div class='stat-value'>{_html_escape(value)}</div></div>"
        )
    return (
        f"<section><div class='stats-grid'>{''.join(items)}</div></section>"
        if items
        else ""
    )


def _render_link_list(title: str, mapping: dict[str, Any]) -> str:
    if not mapping:
        return _render_section(title, "<p>No links.</p>", open_by_default=False)
    items = []
    for key, value in mapping.items():
        href = _path_to_href(value)
        if href:
            items.append(
                f"<li><strong>{_html_escape(key)}</strong>: <a href='{_html_escape(href)}'>{_html_escape(value)}</a></li>"
            )
        else:
            items.append(
                f"<li><strong>{_html_escape(key)}</strong>: {_html_escape(value)}</li>"
            )
    return _render_section(title, f"<ul>{''.join(items)}</ul>", open_by_default=False)


def _render_key_value_table(title: str, mapping: dict[str, Any]) -> str:
    if not mapping:
        return _render_section(title, "<p>No data.</p>")
    rows = []
    for key, value in mapping.items():
        rows.append(
            f"<tr><th>{_html_escape(key)}</th><td>{_html_escape(value)}</td></tr>"
        )
    return _render_section(title, f"<table>{''.join(rows)}</table>")


def _render_rows_table(
    title: str, rows: list[dict[str, Any]], columns: list[str]
) -> str:
    if not rows:
        return _render_section(title, "<p>No rows.</p>", open_by_default=False)
    thead = "".join(f"<th>{_html_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        body = "".join(
            f"<td>{_html_escape(row.get(column, ''))}</td>" for column in columns
        )
        body_rows.append(f"<tr>{body}</tr>")
    return _render_section(
        title,
        f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>",
        open_by_default=False,
    )


def _render_cycle_history_table(title: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return _render_section(
            title, "<p>No recent cycle history.</p>", open_by_default=False
        )
    columns = [
        "cycle",
        "returncode",
        "duration_seconds",
        "guardrail_phase",
        "guardrail_mode",
        "submission_board_items",
        "rewrite_board_items",
        "shortlist_items",
        "todo_closure_rate",
        "todo_backlog",
        "todo_p0_backlog",
    ]
    thead = "".join(f"<th>{_html_escape(column)}</th>" for column in columns)
    body_rows = []
    for row in rows:
        views = row.get("views") or {}
        feedback = row.get("active_source_feedback") or {}
        flattened = {
            "cycle": row.get("cycle"),
            "returncode": row.get("returncode"),
            "duration_seconds": row.get("duration_seconds"),
            "guardrail_phase": row.get("guardrail_phase"),
            "guardrail_mode": row.get("guardrail_mode"),
            "submission_board_items": views.get("submission_board_items"),
            "rewrite_board_items": views.get("rewrite_board_items"),
            "shortlist_items": views.get("shortlist_items"),
            "todo_closure_rate": feedback.get("avg_experiment_todo_closure_rate"),
            "todo_backlog": feedback.get("avg_experiment_todo"),
            "todo_p0_backlog": feedback.get("avg_experiment_todo_p0"),
        }
        row_class = (
            "danger-row"
            if row.get("returncode") not in (0, None)
            else ("warn-row" if (views.get("submission_board_items") or 0) == 0 else "")
        )
        body = "".join(
            f"<td>{_html_escape(flattened.get(column, ''))}</td>" for column in columns
        )
        body_rows.append(f"<tr class='{row_class}'>{body}</tr>")
    content = f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    return _render_section(title, content, open_by_default=False)


def _build_live_dashboard_html(payload: dict[str, Any]) -> str:
    status = payload.get("daemon_status") or {}
    cycle = payload.get("cycle_summary") or {}
    daily = payload.get("daily_summary") or {}
    brief = payload.get("operator_brief") or {}
    runtime_rows = (payload.get("source_runtime_board") or {}).get("rows") or []
    health_rows = (payload.get("source_health_board") or {}).get("rows") or []
    batch_plan_rows = (payload.get("source_batch_plan") or {}).get("rows") or []
    next_batch_rows = (payload.get("source_next_batch") or {}).get("slots") or []
    followup_items = (payload.get("rewrite_followup") or {}).get("items") or []
    trend_metrics = payload.get("trend_metrics") or {}
    active_feedback = status.get("active_source_feedback_snapshot") or {}
    refresh_seconds = payload.get("refresh_seconds") or 30

    style = """
body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #0b1020; color: #e8ecf1; }
a { color: #9fc2ff; }
h1, h2 { color: #ffffff; margin-top: 0; }
details.panel { background: #121936; border: 1px solid #273056; border-radius: 12px; padding: 0; margin-bottom: 16px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); overflow: hidden; }
details.panel > summary { cursor: pointer; list-style: none; padding: 14px 16px; font-weight: 700; color: #fff; background: #111833; }
details.panel > summary::-webkit-details-marker { display: none; }
.panel-body { padding: 16px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px; border-bottom: 1px solid #2a355f; vertical-align: top; }
th { color: #9fb3ff; width: 220px; }
ul { margin: 0; padding-left: 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 18px; }
.stat-card, .trend-card { background: #121936; border: 1px solid #273056; border-radius: 12px; padding: 14px; box-shadow: 0 8px 24px rgba(0,0,0,0.15); }
.stat-label, .trend-title { color: #9aa6c0; font-size: 12px; margin-bottom: 6px; }
.stat-value, .trend-latest { color: #ffffff; font-size: 24px; font-weight: 700; }
.small { color: #9aa6c0; font-size: 12px; }
.badges { margin: 10px 0 18px 0; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #24305c; color: #dbe3ff; margin-right: 8px; margin-bottom: 8px; }
.hero { margin-bottom: 18px; }
.hero p { margin: 6px 0; }
.trend-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 18px; }
.danger-row td { background: rgba(180, 60, 60, 0.18); }
.warn-row td { background: rgba(180, 140, 60, 0.14); }
"""

    hero = _render_stat_cards(
        [
            (
                "Health",
                f"{(cycle.get('health') or brief.get('health') or {}).get('score', 'n/a')}",
            ),
            ("Phase", status.get("guardrail_phase") or "n/a"),
            ("Mode", status.get("guardrail_mode") or "n/a"),
            ("Cycle", status.get("cycle") or 0),
            ("Submissions", (cycle.get("submission_board_items") or 0)),
            ("Rewrite Targets", (cycle.get("rewrite_board_items") or 0)),
            ("Follow-up Δ", (cycle.get("followup_avg_priority_delta") or 0)),
        ]
    )

    trend_cards = (
        "<section><div class='trend-grid'>"
        + "".join(
            [
                _render_trend_card(
                    "Submission Board",
                    trend_metrics.get("submission_board_items") or [],
                    latest=cycle.get("submission_board_items"),
                ),
                _render_trend_card(
                    "Rewrite Board",
                    trend_metrics.get("rewrite_board_items") or [],
                    latest=cycle.get("rewrite_board_items"),
                ),
                _render_trend_card(
                    "Shortlist",
                    trend_metrics.get("shortlist_items") or [],
                    latest=cycle.get("shortlist_items"),
                ),
                _render_trend_card(
                    "Cycle Duration",
                    trend_metrics.get("duration_seconds") or [],
                    latest=(
                        (payload.get("cycle_history") or [{}])[-1].get(
                            "duration_seconds"
                        )
                        if payload.get("cycle_history")
                        else None
                    ),
                ),
                _render_trend_card(
                    "Return Code",
                    trend_metrics.get("returncode") or [],
                    latest=status.get("last_returncode"),
                ),
                _render_trend_card(
                    "TODO Closure",
                    trend_metrics.get("experiment_todo_closure_rate") or [],
                    latest=active_feedback.get("avg_experiment_todo_closure_rate"),
                ),
                _render_trend_card(
                    "TODO Backlog",
                    trend_metrics.get("experiment_todo_backlog") or [],
                    latest=active_feedback.get("avg_experiment_todo"),
                ),
            ]
        )
        + "</div></section>"
    )

    artifact_links = _render_link_list(
        "Artifact Links",
        {
            "submission_board": cycle.get("last_views", {}).get("submission_board"),
            "rewrite_board": cycle.get("last_views", {}).get("rewrite_board"),
            "shortlist": cycle.get("last_views", {}).get("shortlist"),
            "source_runtime_board": cycle.get("source_runtime_board"),
            "source_health_board": cycle.get("source_health_board"),
            "source_batch_plan": cycle.get("source_batch_plan"),
            "operator_brief": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_operator_brief.md")
            )
            or None,
            "handoff_report": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_handoff_report.md")
            )
            or None,
            "daily_report": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_daily_report.md")
            )
            or None,
            "report_index": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "reports" / "index.md")
            )
            or None,
            "report_trends": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "reports" / "trends.md")
            )
            or None,
            "primary_action_queue": (
                status.get("daemon_dir")
                and str(
                    Path(status.get("daemon_dir")) / "latest_primary_action_queue.md"
                )
            )
            or None,
            "cycle_summary": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_cycle_summary.md")
            )
            or None,
            "daily_summary": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "latest_daily_summary.md")
            )
            or None,
            "live_dashboard": payload.get("dashboard_url"),
            "daemon_control": (
                status.get("daemon_dir")
                and str(Path(status.get("daemon_dir")) / "daemon_control.json")
            )
            or None,
        },
    )

    sections = [
        _render_key_value_table(
            "Daemon Status",
            {
                "generated_at": payload.get("generated_at"),
                "state": status.get("state"),
                "cycle": status.get("cycle"),
                "guardrail_phase": status.get("guardrail_phase"),
                "guardrail_mode": status.get("guardrail_mode"),
                "active_source": status.get("active_source"),
                "current_daypart": status.get("current_daypart"),
                "last_error": status.get("last_error"),
                "next_cycle_at": status.get("next_cycle_at"),
            },
        ),
        _render_key_value_table("Daemon Control", payload.get("daemon_control") or {}),
        _render_key_value_table(
            "Control Summary", {"summary": (brief.get("control_summary") or [])}
        ),
        _render_key_value_table(
            "Cycle Summary",
            {
                "success_count": cycle.get("success_count"),
                "failure_count": cycle.get("failure_count"),
                "submission_board_items": cycle.get("submission_board_items"),
                "rewrite_board_items": cycle.get("rewrite_board_items"),
                "shortlist_items": cycle.get("shortlist_items"),
                "followup_avg_priority_delta": cycle.get("followup_avg_priority_delta"),
                "health_recommendation": (cycle.get("health") or {}).get(
                    "recommendation"
                ),
            },
        ),
        _render_key_value_table(
            "Daily Summary",
            {
                "success_count": daily.get("success_count"),
                "failure_count": daily.get("failure_count"),
                "submission_board_items": daily.get("submission_board_items"),
                "rewrite_board_items": daily.get("rewrite_board_items"),
                "shortlist_items": daily.get("shortlist_items"),
            },
        ),
        _render_key_value_table(
            "Daily Report",
            {
                "report_date": (payload.get("daily_report") or {}).get("report_date"),
                "health_state": (payload.get("daily_report") or {}).get("health_state"),
                "top_submission": (
                    (
                        (payload.get("daily_report") or {}).get(
                            "top_submission_targets"
                        )
                        or [{}]
                    )[0].get("name")
                    if (payload.get("daily_report") or {}).get("top_submission_targets")
                    else None
                ),
                "top_rewrite": (
                    (
                        (payload.get("daily_report") or {}).get("top_rewrite_targets")
                        or [{}]
                    )[0].get("name")
                    if (payload.get("daily_report") or {}).get("top_rewrite_targets")
                    else None
                ),
            },
        ),
        _render_key_value_table(
            "Report Archive Index",
            {
                "daily": ((payload.get("report_index") or {}).get("counts") or {}).get(
                    "daily"
                ),
                "handoff": (
                    (payload.get("report_index") or {}).get("counts") or {}
                ).get("handoff"),
            },
        ),
        _render_key_value_table(
            "Report Trends",
            {
                "average_daily_health_score": (payload.get("report_trends") or {}).get(
                    "average_daily_health_score"
                ),
                "daily_health_delta": (payload.get("report_trends") or {}).get(
                    "daily_health_delta"
                ),
                "latest_daily_health_score": (payload.get("report_trends") or {}).get(
                    "latest_daily_health_score"
                ),
                "trend_action_label": (payload.get("report_trends") or {}).get(
                    "trend_action_label"
                ),
                "trend_action_reason": (payload.get("report_trends") or {}).get(
                    "trend_action_reason"
                ),
                "trend_action_command": (payload.get("report_trends") or {}).get(
                    "trend_action_command"
                ),
                "average_todo_closure_rate": (payload.get("report_trends") or {}).get(
                    "average_todo_closure_rate"
                ),
                "latest_todo_closure_rate": (payload.get("report_trends") or {}).get(
                    "latest_todo_closure_rate"
                ),
                "todo_closure_delta": (payload.get("report_trends") or {}).get(
                    "todo_closure_delta"
                ),
                "average_todo_backlog": (payload.get("report_trends") or {}).get(
                    "average_todo_backlog"
                ),
            },
        ),
        _render_key_value_table(
            "Do Now",
            {
                "actions": [
                    f"[{item.get('tier')}] {item.get('source')} :: {item.get('recommendation')}"
                    + (f" | {item.get('command')}" if item.get("command") else "")
                    for item in (brief.get("do_now_actions") or [])
                ]
            },
        ),
        _render_key_value_table(
            "Operator Priorities",
            {
                "priorities": brief.get("priorities"),
                "actions": brief.get("actions"),
                "blockers": brief.get("blockers"),
            },
        ),
        _render_rows_table(
            "Primary Action Queue",
            (payload.get("primary_action_queue") or {}).get("items") or [],
            ["priority", "label", "category", "source", "reason", "command"],
        ),
        _render_key_value_table(
            "Recommended Commands", brief.get("recommended_commands") or {}
        ),
        _render_key_value_table(
            "Handoff Snapshot",
            {
                "attention_label": (payload.get("handoff_report") or {}).get(
                    "attention_label"
                ),
                "recovery_reason": (payload.get("handoff_report") or {}).get(
                    "recovery_reason"
                ),
                "recovery_command": (payload.get("handoff_report") or {}).get(
                    "recovery_command"
                ),
                "health_state": (payload.get("handoff_report") or {}).get(
                    "health_state"
                ),
                "phase": (payload.get("handoff_report") or {}).get("phase"),
                "mode": (payload.get("handoff_report") or {}).get("mode"),
            },
        ),
        _render_rows_table(
            "Source Advisory",
            brief.get("source_advisory") or [],
            ["tier", "source", "state", "health_score", "recommendation", "command"],
        ),
        _render_key_value_table(
            "Source Mix",
            {
                "desired_policy": (brief.get("source_mix_advisory") or {}).get(
                    "desired_policy"
                ),
                "dominant_archetype": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("dominant_archetype"),
                "dominant_workflow_mode": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("dominant_workflow_mode"),
                "archetype_counts": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("archetype_counts"),
                "workflow_mode_counts": (
                    (brief.get("source_mix_advisory") or {}).get("summary") or {}
                ).get("workflow_mode_counts"),
                "recommendations": [
                    item.get("recommendation")
                    for item in (
                        (brief.get("source_mix_advisory") or {}).get("recommendations")
                        or []
                    )
                ],
            },
        ),
        artifact_links,
        _render_rows_table(
            "Source Runtime Board",
            runtime_rows,
            [
                "name",
                "availability_state",
                "availability_reason",
                "priority",
                "current_daypart",
                "target_venue",
                "paper_types",
                "num_ideas",
                "cycles_today",
                "successes_today",
                "cooldown_until_cycle",
                "suggested_action",
            ],
        ),
        _render_rows_table(
            "Source Health Board",
            health_rows,
            [
                "name",
                "health_score",
                "availability_state",
                "priority",
                "current_daypart",
                "target_venue",
                "cycles_today",
                "successes_today",
                "suggested_action",
            ],
        ),
        _render_rows_table(
            "Source Batch Plan",
            batch_plan_rows,
            [
                "tier",
                "source",
                "availability_state",
                "resolved_workflow_mode",
                "source_archetype",
                "batch_profile",
                "workflow_alignment_score",
                "health_score",
                "recommendation",
            ],
        ),
        _render_rows_table(
            "Next Batch Recipe",
            next_batch_rows,
            [
                "lane",
                "source",
                "availability_state",
                "source_workflow_mode",
                "source_archetype",
                "source_batch_profile",
                "share",
                "health_score",
                "rationale",
            ],
        ),
        _render_rows_table(
            "Latest Rewrite Follow-up",
            followup_items,
            [
                "paper",
                "status",
                "priority_before",
                "submission_priority_score",
                "priority_delta",
                "quality_gate_passed",
            ],
        ),
        _render_rows_table(
            "Recent Control Events",
            payload.get("control_history") or [],
            ["timestamp", "type", "matched_key"],
        ),
        _render_rows_table(
            "Recent Failure Hotspots",
            brief.get("failure_hotspots") or [],
            ["reason", "count"],
        ),
        _render_rows_table(
            "Rewrite Style Hotspots",
            brief.get("rewrite_style_hotspots") or [],
            ["style", "score"],
        ),
        _render_cycle_history_table(
            "Recent Cycle History", payload.get("cycle_history") or []
        ),
    ]

    return f"""<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <meta name='viewport' content='width=device-width, initial-scale=1'>
  <meta http-equiv='refresh' content='{_html_escape(refresh_seconds)}'>
  <title>XScientist Daemon Dashboard</title>
  <style>{style}</style>
</head>
<body>
  <div class='hero'>
    <h1>XScientist Daemon Dashboard</h1>
    <p class='small'>Generated at {_html_escape(payload.get('generated_at'))} • auto-refresh every {_html_escape(refresh_seconds)}s</p>
    <div class='badges'>
      <span class='badge'>health={_html_escape((cycle.get('health') or brief.get('health') or {}).get('state', 'n/a'))}</span>
      <span class='badge'>phase={_html_escape(status.get('guardrail_phase'))}</span>
      <span class='badge'>mode={_html_escape(status.get('guardrail_mode'))}</span>
      <span class='badge'>daypart={_html_escape(status.get('current_daypart'))}</span>
      <span class='badge'>paused={_html_escape((status.get('control') or {}).get('paused'))}</span>
    </div>
  </div>
  {hero}
  {trend_cards}
  <div class='grid'>
    {''.join(sections)}
  </div>
</body>
</html>
"""
