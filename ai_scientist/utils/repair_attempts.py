from __future__ import annotations

"""Per-issue repair-attempt history (GEPA-style trajectory log)."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    append_jsonl_artifact,
    artifact_path,
    load_jsonl_artifact,
    save_jsonl_artifact,
    update_pipeline_artifact,
)


_ARTIFACT_NAME = "repair_attempts"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _attempt_path(project_root: str | Path) -> Path:
    return artifact_path(project_root, _ARTIFACT_NAME)


def record_repair_attempt(
    project_root: str | Path,
    issue_id: str,
    *,
    round_index: int,
    job_id: str | None = None,
    status: str = "unknown",
    addressed: bool = False,
    coverage_ratio: float | None = None,
    scores_delta: dict[str, float] | None = None,
    scores_before: dict[str, float] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    issue_id_clean = str(issue_id or "").strip()
    if not issue_id_clean:
        return {"status": "skipped_missing_issue_id"}
    row: dict[str, Any] = {
        "issue_id": issue_id_clean,
        "round_index": int(round_index),
        "job_id": str(job_id or "").strip() or None,
        "status": str(status or "unknown"),
        "addressed": bool(addressed),
        "coverage_ratio": float(coverage_ratio) if coverage_ratio is not None else None,
        "scores_delta": dict(scores_delta or {}),
        "scores_before": dict(scores_before or {}),
        "notes": str(notes or "").strip() or None,
        "generated_at": _now_iso(),
    }
    path = _attempt_path(project_root)
    append_jsonl_artifact(path, row)
    update_pipeline_artifact(
        project_root,
        _ARTIFACT_NAME,
        status="ready",
        producer="repair_attempts",
        depends_on=["review_state"],
    )
    return row


def _load_rows(project_root: str | Path) -> list[dict[str, Any]]:
    path = _attempt_path(project_root)
    return load_jsonl_artifact(path)


def load_attempts_for_issue(
    project_root: str | Path,
    issue_id: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    issue_id_clean = str(issue_id or "").strip()
    if not issue_id_clean:
        return []
    rows = [
        row for row in _load_rows(project_root)
        if str(row.get("issue_id") or "").strip() == issue_id_clean
    ]
    deduped = _dedupe_by_latest(rows)
    deduped.sort(key=lambda r: (int(r.get("round_index") or 0), str(r.get("generated_at") or "")))
    if limit is not None and limit >= 0:
        return deduped[-limit:]
    return deduped


def load_all_attempts(project_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    rows = _load_rows(project_root)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        issue_id = str(row.get("issue_id") or "").strip()
        if not issue_id:
            continue
        grouped.setdefault(issue_id, []).append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for issue_id, items in grouped.items():
        deduped = _dedupe_by_latest(items)
        deduped.sort(key=lambda r: (int(r.get("round_index") or 0), str(r.get("generated_at") or "")))
        out[issue_id] = deduped
    return out


def _dedupe_by_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("issue_id") or "").strip(),
            int(row.get("round_index") or 0),
            str(row.get("job_id") or "").strip(),
        )
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = row
            continue
        if str(row.get("generated_at") or "") >= str(existing.get("generated_at") or ""):
            by_key[key] = row
    return list(by_key.values())


def compact_attempts(project_root: str | Path) -> int:
    """Rewrite the jsonl with deduped rows. Returns the number of rows kept."""
    rows = _load_rows(project_root)
    deduped = _dedupe_by_latest(rows)
    deduped.sort(
        key=lambda r: (
            str(r.get("issue_id") or ""),
            int(r.get("round_index") or 0),
            str(r.get("generated_at") or ""),
        )
    )
    save_jsonl_artifact(_attempt_path(project_root), deduped)
    return len(deduped)


def backfill_scores_delta(
    project_root: str | Path,
    *,
    current_scores: dict[str, float],
) -> int:
    """Fill `scores_delta` on attempts whose `scores_before` was captured but delta is empty.

    Called after a new review round produces fresh aggregated scores. For each attempt
    that has a `scores_before` snapshot and an empty `scores_delta`, write delta =
    current_scores - scores_before for the dims present in both.

    Returns the number of rows updated.
    """
    path = _attempt_path(project_root)
    rows = _load_rows(project_root)
    if not rows or not isinstance(current_scores, dict) or not current_scores:
        return 0
    updated = 0
    for row in rows:
        scores_before = row.get("scores_before")
        existing_delta = row.get("scores_delta") or {}
        if not isinstance(scores_before, dict) or not scores_before:
            continue
        if isinstance(existing_delta, dict) and existing_delta:
            continue
        delta: dict[str, float] = {}
        for key, before_raw in scores_before.items():
            after_raw = current_scores.get(key)
            if before_raw is None or after_raw is None:
                continue
            try:
                delta[key] = round(float(after_raw) - float(before_raw), 4)
            except (TypeError, ValueError):
                continue
        if not delta:
            continue
        row["scores_delta"] = delta
        updated += 1
    if updated:
        save_jsonl_artifact(path, rows)
        update_pipeline_artifact(
            project_root,
            _ARTIFACT_NAME,
            status="ready",
            producer="repair_attempts",
            depends_on=["review_state"],
        )
    return updated
