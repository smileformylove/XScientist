"""Structured writing audit helpers for LaTeX paper refinement loops."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from ai_scientist.llm import extract_json_between_markers, get_response_from_llm


def _coerce_issues(raw_issues: Any, max_issues: int = 6) -> List[Dict[str, str]]:
    if not isinstance(raw_issues, list):
        return []

    issues: List[Dict[str, str]] = []
    for item in raw_issues:
        if not isinstance(item, dict):
            continue
        issue = {
            "category": str(item.get("category", "general")),
            "severity": str(item.get("severity", "medium")),
            "problem": str(item.get("problem", "")),
            "evidence": str(item.get("evidence", "")),
            "action": str(item.get("action", "")),
        }
        if issue["problem"].strip():
            issues.append(issue)
        if len(issues) >= max_issues:
            break
    return issues


def _coerce_checklist_gaps(
    raw_gaps: Any, max_gaps: int = 6
) -> List[Dict[str, str]]:
    if not isinstance(raw_gaps, list):
        return []

    gaps: List[Dict[str, str]] = []
    for item in raw_gaps:
        if not isinstance(item, dict):
            continue
        gap = {
            "item": str(item.get("item", "")).strip(),
            "gap": str(item.get("gap", "")).strip(),
            "action": str(item.get("action", "")).strip(),
        }
        if gap["item"] or gap["gap"]:
            gaps.append(gap)
        if len(gaps) >= max_gaps:
            break
    return gaps


def _coerce_text_list(raw: Any, max_items: int = 8) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    for item in raw:
        text = str(item).strip()
        if text:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _parse_audit_json(response_text: str) -> Dict[str, Any] | None:
    parsed = extract_json_between_markers(response_text)
    if isinstance(parsed, dict):
        return parsed

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        return None
    try:
        parsed_fallback = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed_fallback if isinstance(parsed_fallback, dict) else None


def run_writing_audit(
    *,
    idea_text: str,
    summaries_json: str,
    current_latex: str,
    client: Any,
    model: str,
    system_message: str,
    profile_guidance: str,
    profile_self_checks: str,
    venue_checklist: str = "",
    citation_integrity_rules: str = "",
    humanizer_style_checks: str = "",
    max_issues: int = 6,
) -> Dict[str, Any]:
    """Run one structured writing audit and return normalized findings."""
    prompt = f"""
You are performing a strict writing QA audit for a scientific paper draft.

Draft context:
- Idea description:
```markdown
{idea_text[:6000]}
```

- Experiment summaries:
```json
{summaries_json[:10000]}
```

- Current LaTeX draft:
```latex
{current_latex[:14000]}
```

Profile guidance:
{profile_guidance}

Profile self-checks:
{profile_self_checks}

Venue checklist guidance:
{venue_checklist}

Citation integrity rules:
{citation_integrity_rules}

Humanizer style checks:
{humanizer_style_checks}

Return ONLY JSON in this format:
```json
{{
  "summary": "one short paragraph",
  "issues": [
    {{
      "category": "logic|evidence|latex|figures|citations|clarity|structure",
      "severity": "high|medium|low",
      "problem": "what is wrong",
      "evidence": "concrete snippet or location",
      "action": "specific fix instruction"
    }}
  ],
  "checklist_gaps": [
    {{
      "item": "checklist item name",
      "gap": "what is missing or weak",
      "action": "specific fix"
    }}
  ],
  "citation_risks": ["risk 1", "risk 2"],
  "ai_style_signals": ["signal 1", "signal 2"]
}}
```

Rules:
- Report at most {max_issues} issues.
- Focus on high-impact, actionable issues.
- If no meaningful issue exists, return "issues": [] and a concise summary.
"""

    response, _ = get_response_from_llm(
        prompt=prompt,
        client=client,
        model=model,
        system_message=system_message,
        print_debug=False,
        temperature=0.2,
    )

    parsed = _parse_audit_json(response)
    if not parsed:
        return {
            "summary": "Audit response was not parseable as structured JSON.",
            "issues": [],
            "raw_response_excerpt": response[:1200],
        }

    issues = _coerce_issues(parsed.get("issues"), max_issues=max_issues)
    checklist_gaps = _coerce_checklist_gaps(parsed.get("checklist_gaps"))
    citation_risks = _coerce_text_list(parsed.get("citation_risks"))
    ai_style_signals = _coerce_text_list(parsed.get("ai_style_signals"))
    summary = str(parsed.get("summary", "")).strip() or (
        "No structured summary was returned by the audit."
    )
    return {
        "summary": summary,
        "issues": issues,
        "checklist_gaps": checklist_gaps,
        "citation_risks": citation_risks,
        "ai_style_signals": ai_style_signals,
        "raw_response_excerpt": response[:1200],
    }


def format_writing_audit_for_prompt(audit_result: Dict[str, Any]) -> str:
    """Render audit findings into compact text for reflection prompts."""
    summary = str(audit_result.get("summary", "")).strip()
    issues = audit_result.get("issues")
    checklist_gaps = audit_result.get("checklist_gaps")
    citation_risks = audit_result.get("citation_risks")
    ai_style_signals = audit_result.get("ai_style_signals")

    lines: List[str] = []
    if summary:
        lines.append(f"Audit summary: {summary}")

    if isinstance(issues, list) and issues:
        lines.append("Top audit issues:")
        for idx, issue in enumerate(issues, start=1):
            if not isinstance(issue, dict):
                continue
            lines.append(
                f"{idx}. [{issue.get('severity', 'medium')}/{issue.get('category', 'general')}] "
                f"problem={issue.get('problem', '')} | evidence={issue.get('evidence', '')} | "
                f"action={issue.get('action', '')}"
            )
    else:
        lines.append("Top audit issues: none")

    if isinstance(checklist_gaps, list) and checklist_gaps:
        lines.append("Checklist gaps:")
        for idx, gap in enumerate(checklist_gaps, start=1):
            if not isinstance(gap, dict):
                continue
            lines.append(
                f"{idx}. item={gap.get('item', '')} | gap={gap.get('gap', '')} | "
                f"action={gap.get('action', '')}"
            )
    else:
        lines.append("Checklist gaps: none")

    if isinstance(citation_risks, list) and citation_risks:
        lines.append("Citation risks:")
        for idx, risk in enumerate(citation_risks, start=1):
            lines.append(f"{idx}. {risk}")
    else:
        lines.append("Citation risks: none")

    if isinstance(ai_style_signals, list) and ai_style_signals:
        lines.append("AI-style signals:")
        for idx, signal in enumerate(ai_style_signals, start=1):
            lines.append(f"{idx}. {signal}")
    else:
        lines.append("AI-style signals: none")

    return "\n".join(lines)
