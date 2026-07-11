"""Shared workflow glue for deterministic manuscript integrity forensics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ai_scientist.utils.integrity_forensics import run_integrity_forensics


def find_integrity_latex_sources(root: str | Path) -> list[Path]:
    root_path = Path(root)
    candidates = [root_path / "latex" / "template.tex", root_path / "template.tex"]
    return [path for path in candidates if path.exists()]


def run_integrity_forensics_for_manuscript(
    *,
    root: str | Path,
    paper_id: str,
    enabled: bool,
    latex_paths: Sequence[str | Path] | None = None,
    observability_level: int = 1,
) -> dict[str, Any]:
    """Run deterministic checks and return a pipeline-safe result payload."""

    if not enabled:
        return {"enabled": False, "status": "disabled"}
    tex_sources = (
        find_integrity_latex_sources(root)
        if latex_paths is None
        else [Path(path) for path in latex_paths]
    )
    if not tex_sources:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": "no LaTeX source found for integrity forensics",
        }
    out_dir = Path(root) / "integrity_forensics"
    try:
        result = run_integrity_forensics(
            paper_id=paper_id,
            latex_paths=tex_sources,
            output_dir=out_dir,
            observability_level=observability_level,
        )
    except Exception as exc:  # noqa: BLE001 - keep the owning pipeline alive
        return {
            "enabled": True,
            "status": "error",
            "error": str(exc),
            "output_dir": str(out_dir),
        }
    report = dict(result.report)
    report["report_path"] = result.files.get("report")
    report["markdown_path"] = result.files.get("markdown")
    return {
        "enabled": True,
        "status": "completed",
        "output_dir": str(out_dir),
        "report": report,
        "files": dict(result.files),
        "overall_verdict": report.get("overall_verdict"),
        "finding_count": (report.get("counts") or {}).get("findings"),
    }


def summarize_integrity_forensics_result(result: dict[str, Any]) -> dict[str, Any]:
    nested = (
        result.get("integrity_forensics")
        if isinstance(result.get("integrity_forensics"), dict)
        else {}
    )
    files = nested.get("files") if isinstance(nested.get("files"), dict) else {}
    return {
        "enabled": (
            result.get("integrity_forensics_enabled")
            if result.get("integrity_forensics_enabled") is not None
            else nested.get("enabled")
        ),
        "status": (
            result.get("integrity_forensics_status")
            or nested.get("status")
            or "unknown"
        ),
        "verdict": (
            result.get("integrity_forensics_verdict")
            or nested.get("overall_verdict")
            or "unknown"
        ),
        "findings": (
            result.get("integrity_forensics_findings")
            if result.get("integrity_forensics_findings") is not None
            else nested.get("finding_count")
        ),
        "report_file": (
            result.get("integrity_forensics_report_file")
            or files.get("report")
        ),
        "markdown_file": (
            result.get("integrity_forensics_markdown_file")
            or files.get("markdown")
        ),
    }


def integrity_forensics_result_fields(result: dict[str, Any]) -> dict[str, Any]:
    summary = summarize_integrity_forensics_result({"integrity_forensics": result})
    return {
        "integrity_forensics_enabled": summary.get("enabled"),
        "integrity_forensics_status": summary.get("status"),
        "integrity_forensics_verdict": summary.get("verdict"),
        "integrity_forensics_findings": summary.get("findings"),
        "integrity_forensics_report_file": summary.get("report_file"),
        "integrity_forensics_markdown_file": summary.get("markdown_file"),
    }


def integrity_report_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    report = result.get("report") if isinstance(result, dict) else None
    return report if isinstance(report, dict) else None
