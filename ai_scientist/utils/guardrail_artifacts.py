"""Helpers for loading writeup guardrail artifacts from experiment folders."""

from __future__ import annotations

import json
import os.path as osp
from typing import Any, Dict, List, Optional, Tuple

from ai_scientist.writeup_guardrails import (
    has_blocking_guardrail_violations,
    list_blocking_guardrail_reasons,
)


def load_guardrail_artifacts(exp_dir: str) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Load final guardrail findings and blocking reasons for one experiment."""
    findings_path = osp.join(exp_dir, "writing_audits", "final_guardrail_report.json")
    reasons_path = osp.join(
        exp_dir, "writing_audits", "final_guardrail_failure_reasons.json"
    )

    findings: Optional[Dict[str, Any]] = None
    reasons: List[str] = []

    if osp.exists(findings_path):
        try:
            with open(findings_path, "r", encoding="utf-8") as f_findings:
                loaded = json.load(f_findings)
            if isinstance(loaded, dict):
                # Prefer explicit nested findings when present.
                if isinstance(loaded.get("findings"), dict):
                    findings = loaded.get("findings")
                else:
                    findings = loaded
        except Exception:
            findings = None

    if osp.exists(reasons_path):
        try:
            with open(reasons_path, "r", encoding="utf-8") as f_reasons:
                loaded = json.load(f_reasons)
            if findings is None and isinstance(loaded, dict) and isinstance(
                loaded.get("findings"), dict
            ):
                findings = loaded.get("findings")
            raw_reasons: Any = []
            if isinstance(loaded, dict):
                raw_reasons = loaded.get(
                    "reasons",
                    loaded.get(
                        "guardrail_blocking_reasons",
                        loaded.get("blocking_reasons", []),
                    ),
                )
            elif isinstance(loaded, list):
                raw_reasons = loaded
            elif isinstance(loaded, str):
                raw_reasons = [loaded]

            if isinstance(raw_reasons, str):
                raw_reasons = [raw_reasons]
            if isinstance(raw_reasons, list):
                reasons = [str(item).strip() for item in raw_reasons if str(item).strip()]
        except Exception:
            reasons = []
            try:
                with open(reasons_path, "r", encoding="utf-8") as f_raw_reasons:
                    raw_text = f_raw_reasons.read().strip()
                if raw_text:
                    reasons = [raw_text]
            except Exception:
                reasons = []

    if findings is not None and not reasons and has_blocking_guardrail_violations(
        findings,
        allow_placeholder_citations=False,
        require_venue_sections=True,
    ):
        reasons = list_blocking_guardrail_reasons(
            findings,
            allow_placeholder_citations=False,
            require_venue_sections=True,
        )

    return findings, reasons


def result_passed_writeup_guardrails(result: Dict[str, Any]) -> bool:
    """Infer whether a pipeline result made it past strict writeup guardrails."""
    if result.get("status") == "success":
        return True
    stage = str(result.get("stage") or "").strip().lower()
    return bool(stage) and stage != "writeup"
