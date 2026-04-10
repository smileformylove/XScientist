"""Reusable guardrails for writeup quality and submission readiness.

The rules are inspired by public academic-writing skill libraries:
- venue-oriented submission checklists
- citation hallucination prevention
- anti-pattern checks for overly AI-sounding prose
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set


_VENUE_ALIASES = {
    "neurips": "neurips",
    "nips": "neurips",
    "icml": "icml",
    "iclr": "iclr",
    "acl": "acl",
    "cvpr": "cvpr",
    "aaai": "aaai",
    "colm": "colm",
    "journal": "journal",
    "nature": "nature",
}


_VENUE_CHECKLISTS: Dict[str, List[str]] = {
    "neurips": [
        "Claims in abstract/introduction must match reported evidence and limits.",
        "Include a dedicated limitations discussion with concrete failure modes.",
        "Document reproducibility details: data splits, hyperparameters, seeds.",
        "Report uncertainty: error bars/confidence intervals/statistical method.",
        "State compute resources (GPU type/count, runtime, approximate budget).",
        "Describe data/code access and exact reproduction instructions.",
        "Address ethics/broader impacts when relevant to claims or deployment.",
    ],
    "icml": [
        "Broader impact statement should be explicit and concrete.",
        "Reproducibility details must include splits, search ranges, selection logic.",
        "All key comparisons should include uncertainty reporting.",
        "Avoid de-anonymization leaks in blind-submission mode.",
        "Summarize compute budget and implementation details for replication.",
    ],
    "iclr": [
        "Claims should be falsifiable and directly tied to measured evidence.",
        "Add reproducibility statement: code/data/checkpoint availability plan.",
        "Include limitations and potential risks for real-world usage.",
        "If LLM usage is substantial, include an explicit disclosure note.",
        "Avoid over-claiming generalization beyond tested regimes.",
    ],
    "acl": [
        "Limitations section is mandatory for NLP-focused submissions.",
        "Discuss bias/fairness and dual-use concerns where applicable.",
        "Provide data provenance and preprocessing transparency.",
        "Ensure claim wording matches actually reported experiments.",
        "Include reproducibility details and implementation constraints.",
    ],
    "cvpr": [
        "Prioritize visual clarity: figures/captions must stand alone.",
        "Detail dataset splits, preprocessing, augmentations, and metrics.",
        "Use fair and strong baselines with reproducible settings.",
        "Report robustness/failure cases, not only best-case outcomes.",
        "State compute/training setup for comparability.",
    ],
    "journal": [
        "Strengthen motivation and scope definition with explicit boundaries.",
        "Provide thorough ablations and error analysis, not only headline metrics.",
        "Detail reproducibility and long-term artifact availability.",
        "State limitations, risks, and assumptions comprehensively.",
        "Ensure references are precise and historically grounded.",
    ],
    "nature": [
        "Highlight one central contribution with broad-impact motivation.",
        "Use conservative claims and clear evidence chain for each claim.",
        "Explicitly discuss limitations, uncertainty, and reproducibility.",
        "Emphasize significance without marketing language.",
        "Keep narrative concise and evidence-dense.",
    ],
}

_VENUE_REQUIRED_SECTION_PATTERNS: Dict[str, List[List[str]]] = {
    "neurips": [["Limitations", "limitations"], ["Broader Impact", "impact statement", "broader impact"]],
    "icml": [["Broader Impact", "broader impact", "societal impact"], ["Limitations", "limitations"]],
    "iclr": [["Limitations", "limitations"]],
    "acl": [["Limitations", "limitations"]],
    "cvpr": [["Limitations", "limitations"]],
    "journal": [["Limitations", "limitations"], ["Ethics", "ethics", "broader impact"]],
    "nature": [["Limitations", "limitations"], ["Significance", "significance", "broader impact"]],
}

_AI_STYLE_PATTERNS: Dict[str, str] = {
    "promotional_groundbreaking": r"\bgroundbreaking\b",
    "promotional_pivotal": r"\bpivotal\b",
    "promotional_vibrant": r"\bvibrant\b",
    "inflated_testament": r"\btestament\b",
    "inflated_underscores": r"\bunderscore(?:s|d)?\b",
    "vague_experts": r"\bexperts?\s+(argue|say|believe|suggest)\b",
    "vague_some_studies": r"\bsome studies\b",
    "buzzword_landscape": r"\blandscape\b",
    "buzzword_intricate": r"\bintricate\b",
    "formulaic_not_only": r"\bnot only\b",
}


def normalize_venue_name(venue: str | None) -> str | None:
    """Normalize venue aliases to canonical names."""
    if venue is None:
        return None
    normalized = str(venue).strip().lower()
    if not normalized:
        return None
    return _VENUE_ALIASES.get(normalized, normalized)


def render_venue_checklist(venue: str | None, max_items: int = 7) -> str:
    """Render concise venue checklist guidance for prompts."""
    canonical = normalize_venue_name(venue)
    if canonical is None:
        items = [
            "Claims should be falsifiable and supported by direct evidence.",
            "Include limitations, failure modes, and reproducibility details.",
            "Report uncertainty and avoid over-claiming significance.",
            "Provide concrete data/code/compute documentation.",
        ]
        header = "Venue checklist: generic submission quality baseline"
    else:
        items = _VENUE_CHECKLISTS.get(canonical)
        if not items:
            items = [
                "Align claims with evidence and explicit limitations.",
                "Provide reproducibility details and compute disclosure.",
                "Report uncertainty and avoid rhetorical over-claiming.",
            ]
        header = f"Venue checklist ({canonical})"

    lines = [header]
    for idx, item in enumerate(items[:max_items], start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def render_citation_integrity_rules() -> str:
    """Return non-negotiable citation safety rules for prompts."""
    return "\n".join(
        [
            "Citation integrity rules:",
            "1. Never fabricate citations, DOIs, authors, venues, or years.",
            "2. Only cite entries that exist in references.bib or are explicitly marked placeholders.",
            "3. If a citation cannot be verified, keep an explicit placeholder key",
            "   such as \\cite{PLACEHOLDER_author2024_verify_this} and flag it for human verification.",
            "4. Do not convert uncertain claims into definitive citation-backed statements.",
            "5. Preserve existing verified BibTeX entries unless there is a clear correction reason.",
        ]
    )


def render_humanizer_style_rules(profile_name: str | None) -> str:
    """Return style anti-pattern checks inspired by humanizer skill.

    This guidance is strongest when the selected profile explicitly asks for
    humanized prose, but still remains useful as a lightweight check otherwise.
    """
    normalized = (profile_name or "").strip().lower().replace("-", "_")
    is_humanized = normalized in {"humanized_academic", "humanizer"}

    common_checks = [
        "Avoid promotional wording (e.g., 'groundbreaking', 'vibrant', 'pivotal') unless evidence justifies it.",
        "Avoid vague authority phrases ('experts say', 'some studies') without explicit references.",
        "Reduce repetitive sentence templates and overused abstract buzzwords.",
        "Prefer direct verbs and concrete nouns over inflated nominalized phrasing.",
        "Keep tone precise and natural; do not sound like marketing copy.",
    ]
    if is_humanized:
        common_checks.extend(
            [
                "Vary sentence rhythm to avoid monotonous AI-like cadence.",
                "Acknowledge uncertainty explicitly where evidence is mixed.",
                "Preserve technical precision while reducing formulaic filler transitions.",
            ]
        )

    lines = [f"Humanizer style checks (profile={normalized or 'default'}):"]
    for idx, item in enumerate(common_checks, start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines)


def _extract_bib_keys(latex_text: str) -> Set[str]:
    keys: Set[str] = set()
    for match in re.finditer(r"@[\w]+\s*\{\s*([^,\s]+)\s*,", latex_text):
        key = match.group(1).strip()
        if key:
            keys.add(key)
    return keys


def _extract_cite_keys(latex_text: str) -> Set[str]:
    keys: Set[str] = set()
    for match in re.finditer(r"\\cite[a-zA-Z*]*\{([^}]*)\}", latex_text):
        raw_keys = match.group(1).split(",")
        for raw_key in raw_keys:
            key = raw_key.strip()
            if key:
                keys.add(key)
    return keys


def find_missing_bibtex_keys(latex_text: str) -> List[str]:
    """Find citation keys used in text but absent from embedded BibTeX."""
    bib_keys = _extract_bib_keys(latex_text)
    cite_keys = _extract_cite_keys(latex_text)
    missing = sorted(
        key
        for key in cite_keys
        if key not in bib_keys and not key.startswith("PLACEHOLDER_")
    )
    return missing


def find_placeholder_citation_keys(latex_text: str) -> List[str]:
    """Find explicit placeholder citations that still need human verification."""
    cite_keys = _extract_cite_keys(latex_text)
    return sorted(key for key in cite_keys if key.startswith("PLACEHOLDER_"))


def build_citation_consistency_report(latex_text: str, max_items: int = 12) -> str:
    """Build a compact reflection block for citation consistency checks."""
    missing_keys = find_missing_bibtex_keys(latex_text)
    placeholder_keys = find_placeholder_citation_keys(latex_text)

    lines = ["Citation consistency checks:"]
    if missing_keys:
        lines.append(
            "Missing BibTeX entries for cited keys: "
            + ", ".join(missing_keys[:max_items])
        )
        if len(missing_keys) > max_items:
            lines.append(f"... and {len(missing_keys) - max_items} more missing keys.")
    else:
        lines.append("No missing BibTeX entries detected for cited keys.")

    if placeholder_keys:
        lines.append(
            "Placeholder citation keys requiring manual verification: "
            + ", ".join(placeholder_keys[:max_items])
        )
        if len(placeholder_keys) > max_items:
            lines.append(
                f"... and {len(placeholder_keys) - max_items} more placeholders."
            )
    else:
        lines.append("No explicit placeholder citation keys detected.")

    return "\n".join(lines)


def _strip_references_bib_block(latex_text: str) -> str:
    return re.sub(
        r"\\begin{filecontents}{references\.bib}.*?\\end{filecontents}",
        "",
        latex_text,
        flags=re.IGNORECASE | re.DOTALL,
    )


def _has_any_section_alias(latex_text: str, aliases: List[str]) -> bool:
    for alias in aliases:
        section_pattern = (
            r"\\section\*?\{[^}]*" + re.escape(alias) + r"[^}]*\}"
        )
        heading_pattern = (
            r"(^|\n)\s*(?:#+\s*)?"
            + re.escape(alias)
            + r"\s*(?:[:：]|$)"
        )
        if re.search(section_pattern, latex_text, flags=re.IGNORECASE):
            return True
        if re.search(heading_pattern, latex_text, flags=re.IGNORECASE):
            return True
    return False


def find_required_section_gaps(latex_text: str, venue: str | None) -> List[str]:
    """Detect missing venue-required sections from a coarse section-name check."""
    canonical = normalize_venue_name(venue)
    if canonical is None:
        return []

    required = _VENUE_REQUIRED_SECTION_PATTERNS.get(canonical, [])
    if not required:
        return []

    gaps: List[str] = []
    for item in required:
        if not item:
            continue
        display_name = item[0]
        aliases = item[1:] or [display_name]
        if not _has_any_section_alias(latex_text, aliases):
            gaps.append(display_name)
    return gaps


def find_ai_style_markers(latex_text: str, max_items: int = 10) -> List[str]:
    """Identify likely AI-style lexical markers in manuscript body text."""
    body = _strip_references_bib_block(latex_text)
    hits: List[tuple[int, str]] = []
    for label, pattern in _AI_STYLE_PATTERNS.items():
        count = len(re.findall(pattern, body, flags=re.IGNORECASE))
        if count > 0:
            hits.append((count, label))
    hits.sort(key=lambda x: (-x[0], x[1]))
    return [f"{label}:{count}" for count, label in hits[:max_items]]


def collect_guardrail_findings(
    latex_text: str,
    venue: str | None,
    *,
    max_items: int = 12,
) -> Dict[str, Any]:
    """Collect machine-readable guardrail findings for downstream policy checks."""
    missing_bib_keys = find_missing_bibtex_keys(latex_text)
    placeholder_keys = find_placeholder_citation_keys(latex_text)
    missing_sections = find_required_section_gaps(latex_text, venue)
    ai_style_markers = find_ai_style_markers(latex_text, max_items=max_items)
    return {
        "venue": normalize_venue_name(venue),
        "missing_bibtex_keys": missing_bib_keys[:max_items],
        "missing_bibtex_key_count": len(missing_bib_keys),
        "placeholder_citation_keys": placeholder_keys[:max_items],
        "placeholder_citation_key_count": len(placeholder_keys),
        "missing_required_sections": missing_sections[:max_items],
        "missing_required_section_count": len(missing_sections),
        "ai_style_markers": ai_style_markers[:max_items],
        "ai_style_marker_count": len(ai_style_markers),
    }


def build_submission_guardrail_report(
    latex_text: str,
    venue: str | None,
    *,
    max_items: int = 12,
) -> str:
    """Build a compact report for reflection prompts and logs."""
    findings = collect_guardrail_findings(latex_text, venue, max_items=max_items)
    lines = [f"Submission guardrail checks (venue={findings.get('venue') or 'generic'}):"]

    missing = findings.get("missing_bibtex_keys", [])
    if missing:
        lines.append("Missing BibTeX entries for cited keys: " + ", ".join(missing))
    else:
        lines.append("Missing BibTeX entries: none")

    placeholders = findings.get("placeholder_citation_keys", [])
    if placeholders:
        lines.append("Placeholder citation keys present: " + ", ".join(placeholders))
    else:
        lines.append("Placeholder citation keys: none")

    section_gaps = findings.get("missing_required_sections", [])
    if section_gaps:
        lines.append("Potentially missing venue-required sections: " + ", ".join(section_gaps))
    else:
        lines.append("Potentially missing venue-required sections: none")

    style_hits = findings.get("ai_style_markers", [])
    if style_hits:
        lines.append("AI-style lexical markers (heuristic): " + ", ".join(style_hits))
    else:
        lines.append("AI-style lexical markers (heuristic): none")
    return "\n".join(lines)


def has_blocking_guardrail_violations(
    findings: Dict[str, Any],
    *,
    allow_placeholder_citations: bool = False,
    require_venue_sections: bool = True,
) -> bool:
    """Decide whether findings should block strict submission-grade writeup."""
    if int(findings.get("missing_bibtex_key_count", 0)) > 0:
        return True
    if not allow_placeholder_citations and int(
        findings.get("placeholder_citation_key_count", 0)
    ) > 0:
        return True
    if require_venue_sections and int(findings.get("missing_required_section_count", 0)) > 0:
        return True
    return False


def list_blocking_guardrail_reasons(
    findings: Dict[str, Any],
    *,
    allow_placeholder_citations: bool = False,
    require_venue_sections: bool = True,
) -> List[str]:
    """Return concrete human-readable reasons for blocking guardrail checks."""
    reasons: List[str] = []
    missing_keys = int(findings.get("missing_bibtex_key_count", 0))
    placeholder_keys = int(findings.get("placeholder_citation_key_count", 0))
    missing_sections = int(findings.get("missing_required_section_count", 0))

    if missing_keys > 0:
        reasons.append(f"missing_bibtex_keys={missing_keys}")
    if not allow_placeholder_citations and placeholder_keys > 0:
        reasons.append(f"placeholder_citation_keys={placeholder_keys}")
    if require_venue_sections and missing_sections > 0:
        reasons.append(f"missing_required_sections={missing_sections}")
    return reasons


def build_guardrail_repair_plan(
    findings: Dict[str, Any],
    venue: str | None = None,
    *,
    max_items: int = 12,
) -> str:
    """Build an actionable repair plan from machine-readable guardrail findings."""
    canonical_venue = normalize_venue_name(venue) or findings.get("venue")
    missing_bibtex_keys = findings.get("missing_bibtex_keys", [])
    placeholder_keys = findings.get("placeholder_citation_keys", [])
    missing_sections = findings.get("missing_required_sections", [])

    lines: List[str] = [
        f"Guardrail repair plan (venue={canonical_venue or 'generic'}):",
        "1. Keep all claims grounded in existing experiment logs and metrics.",
        "2. Do not fabricate citations or rewrite uncertain claims as verified facts.",
    ]

    if isinstance(missing_bibtex_keys, list) and missing_bibtex_keys:
        subset = [str(item).strip() for item in missing_bibtex_keys[:max_items] if str(item).strip()]
        if subset:
            lines.append(
                "3. Add BibTeX entries for cited keys currently missing in references.bib: "
                + ", ".join(subset)
            )

    if isinstance(placeholder_keys, list) and placeholder_keys:
        subset = [str(item).strip() for item in placeholder_keys[:max_items] if str(item).strip()]
        if subset:
            lines.append(
                "4. Resolve placeholder citations by either replacing them with verified keys or removing unsupported claims: "
                + ", ".join(subset)
            )

    if isinstance(missing_sections, list) and missing_sections:
        subset = [str(item).strip() for item in missing_sections[:max_items] if str(item).strip()]
        if subset:
            lines.append(
                "5. Add venue-required sections with concrete, evidence-backed content: "
                + ", ".join(subset)
            )

    lines.append(
        "6. Return a complete LaTeX file that preserves references.bib consistency with all in-text citations."
    )
    return "\n".join(lines)
