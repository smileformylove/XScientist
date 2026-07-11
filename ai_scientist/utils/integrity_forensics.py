"""Deterministic integrity forensics for generated manuscripts.

This module adapts the reviewer-side evidence-ledger pattern from
``wanshuiyin/Anti-Autoresearch`` into XScientist's native runtime.  The goal is
not authorship detection.  It builds a span-anchored ledger from LaTeX/text,
runs model-free consistency checks, then applies deterministic adjudication
rules so the result can be used by quality gates and ARA consumers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

LEDGER_VERSION = "xscientist.integrity.ledger.v1"
REPORT_VERSION = "xscientist.integrity.report.v1"
ADJUDICATOR_ID = "xscientist-deterministic-rules-v1"

SEVERITY_ORDER = {"info": 0, "minor": 1, "major": 2, "critical": 3}
SEVERITY_NAME = {value: key for key, value in SEVERITY_ORDER.items()}

NUMBER_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(\\?%|percent|points?|pts?|x|×)?")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
CAPTION_RE = re.compile(r"\\caption\*?\{")
SECTION_RE = re.compile(r"\\(?:sub)*section\*?\{([^}]*)\}")
METRIC_RE = re.compile(
    r"\b(accuracy|acc|F1|F-?1|BLEU|ROUGE|exact match|EM|precision|recall|"
    r"AUC|AUROC|AUPRC|perplexity|PPL|latency|throughput|win[- ]rate|"
    r"pass@\d+|mAP|IoU|MSE|RMSE|MAE|success rate|reward|score)\b",
    re.IGNORECASE,
)
SCOPE_RE = re.compile(
    r"\b(comprehensive|extensive|exhaustive|robust|robustly|consistently|"
    r"state[- ]of[- ]the[- ]art|SOTA|outperform\w*|significantly|"
    r"substantially|general(?:ly|izes)?|always|all (?:tasks|datasets|settings)|"
    r"first to)\b",
    re.IGNORECASE,
)

DELTA_PATTERNS = (
    re.compile(
        r"from\s+[^\d.]{0,12}?(\d+(?:\.\d+)?)\s*%?[^.]{0,40}?\bto\s+"
        r"[^\d.]{0,12}?(\d+(?:\.\d+)?)\s*%?[^.]{0,80}?"
        r"(\d+(?:\.\d+)?)\s*%\s*(relative\s+)?"
        r"(gain|improvement|increase|boost|reduction|drop|decrease)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d+(?:\.\d+)?)\s*%\s*(relative\s+)?"
        r"(gain|improvement|increase|boost|reduction|drop|decrease)"
        r"[^.]{0,80}?from\s+[^\d.]{0,12}?(\d+(?:\.\d+)?)\s*%?"
        r"[^.]{0,40}?\bto\s+[^\d.]{0,12}?(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    ),
)
DELTA_TOLERANCE = 0.6

PIPELINE_ARTIFACT_PHRASES = (
    "as an ai language model",
    "as a large language model",
    "as an ai assistant",
    "i'm sorry, but i cannot",
    "i cannot fulfill",
    "i cannot fulfil",
    "i am unable to provide",
    "as of my last knowledge update",
    "regenerate response",
    "<your text here>",
    "[insert ",
    "<insert ",
    "todo: cite",
    "todo: add citation",
    "[citation needed]",
    "lorem ipsum",
)

GRIM_METRIC_RE = re.compile(
    r"\b(accuracy|acc|correct(?:ly)?|success(?:\s*rate)?|error\s*rate|"
    r"exact\s*match|EM|solved|passed|pass\s*rate|win[- ]?rate|"
    r"answered\s+correctly)\b",
    re.IGNORECASE,
)
GRIM_N_RE = re.compile(
    r"N\s*=\s*(\d{1,7})\b"
    r"|\b(\d{1,7})\s*(?:-|\s)?(?:item|example|question|sample|instance|"
    r"case|problem|sentence|image|datapoint|document|prompt|query|trial)s?\b"
    r"|\b(?:test|eval(?:uation)?|held[- ]?out|dev|validation)\s*set\s*of\s*"
    r"(\d{1,7})\b"
    r"|\bout\s+of\s+(\d{1,7})\b",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
GRIM_EXCLUDE_RE = re.compile(
    r"\b(macro|micro|weighted|balanced|harmonic|geometric|improv\w*|"
    r"increase\w*|decreas\w*|reduc\w*|\bgain\b|relative|average\s+of|"
    r"mean\s+(?:over|of|across)|averaged\s+(?:over|across))\b",
    re.IGNORECASE,
)


@dataclass
class IntegrityLedger:
    """Span ledger consumed by integrity checkers."""

    paper_id: str
    observability_level: int
    claims: list[dict[str, Any]]
    source_files: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = ""
    ledger_version: str = LEDGER_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IntegrityRunResult:
    """Artifacts from one deterministic integrity pass."""

    ledger: dict[str, Any]
    findings: list[dict[str, Any]]
    report: dict[str, Any]
    output_dir: str | None = None
    files: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _norm_ws(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _split_sentences(paragraph: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", paragraph)
    return [part.strip() for part in parts if part.strip()]


def _iter_paragraphs(lines: Sequence[str]) -> Iterable[tuple[str, int, int]]:
    buf: list[str] = []
    start: int | None = None
    end: int | None = None
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            if buf and start is not None and end is not None:
                yield " ".join(buf), start, end
            buf = []
            start = None
            end = None
            continue
        if start is None:
            start = idx
        end = idx
        buf.append(line.strip())
    if buf and start is not None and end is not None:
        yield " ".join(buf), start, end


def _section_labeler(lines: Sequence[str]):
    marks: list[tuple[int, str]] = []
    table_idx = 0
    figure_idx = 0
    in_abstract = False
    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        section = SECTION_RE.search(stripped)
        if section:
            marks.append((idx, section.group(1).strip().lower()))
        if "\\begin{abstract}" in stripped:
            marks.append((idx, "abstract"))
            in_abstract = True
        if "\\end{abstract}" in stripped and in_abstract:
            marks.append((idx, "intro"))
            in_abstract = False
        if "\\begin{table" in stripped:
            table_idx += 1
            marks.append((idx, f"table:{table_idx}"))
        if "\\end{table" in stripped:
            marks.append((idx, "body"))
        if "\\begin{figure" in stripped:
            figure_idx += 1
            marks.append((idx, f"figure:{figure_idx}"))
        if "\\end{figure" in stripped:
            marks.append((idx, "body"))
        if re.search(r"\\(?:begin\{appendix\}|appendix)\b", stripped):
            marks.append((idx, "appendix"))

    marks.sort()

    def label(line_no: int) -> str:
        current = "body"
        for mark_line, mark_label in marks:
            if mark_line <= line_no:
                current = mark_label
            else:
                break
        return current

    return label


def _tabular_lines(lines: Sequence[str]) -> set[int]:
    inside: set[int] = set()
    depth = 0
    for idx, line in enumerate(lines, start=1):
        if "\\begin{tabular" in line:
            depth += 1
        if depth > 0:
            inside.add(idx)
        if "\\end{tabular" in line:
            depth = max(0, depth - 1)
    return inside


def _strip_latex_commands_for_sentence(line: str) -> str:
    """Remove structural LaTeX commands while preserving surrounding prose."""

    stripped = line.strip()
    stripped = re.sub(
        r"\\(?:documentclass|usepackage)(?:\[[^\]]*\])?\{[^}]*\}", " ", stripped
    )
    stripped = re.sub(
        r"\\(?:begin|end)\{(?:document|abstract|table|figure|tabular)[^}]*\}",
        " ",
        stripped,
    )
    stripped = re.sub(
        r"\\(?:section|subsection|subsubsection)\*?\{([^}]*)\}", r"\1.", stripped
    )
    return re.sub(r"\s+", " ", stripped).strip()


def _is_block_boundary(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if re.search(r"\\(?:begin|end)\{(?:table|figure|tabular)\*?\}", stripped):
        return True
    if SECTION_RE.search(stripped):
        return True
    return False


def _iter_prose_blocks(
    lines: Sequence[str],
    *,
    tab_lines: set[int],
) -> Iterable[tuple[str, int]]:
    """Yield cleaned prose blocks assembled across physical LaTeX line wraps."""

    buffer: list[str] = []
    start_line: int | None = None

    def flush() -> tuple[str, int] | None:
        nonlocal buffer, start_line
        if not buffer or start_line is None:
            buffer = []
            start_line = None
            return None
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        line = start_line
        buffer = []
        start_line = None
        return (text, line) if text else None

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if line_no in tab_lines or stripped.startswith("%"):
            pending = flush()
            if pending:
                yield pending
            continue
        if SECTION_RE.search(stripped):
            pending = flush()
            if pending:
                yield pending
            cleaned = _strip_latex_commands_for_sentence(line)
            if cleaned:
                yield cleaned, line_no
            continue
        if _is_block_boundary(line):
            pending = flush()
            if pending:
                yield pending
            continue
        cleaned = _strip_latex_commands_for_sentence(line)
        if not cleaned:
            continue
        if start_line is None:
            start_line = line_no
        buffer.append(cleaned)

    pending = flush()
    if pending:
        yield pending


def _parse_number(num: str, unit: str | None) -> dict[str, Any]:
    try:
        normalized = float(num)
    except ValueError:
        normalized = None
    resolved_unit = None
    if unit:
        low = unit.lower()
        if "%" in unit or low == "percent":
            resolved_unit = "%"
        elif low.startswith(("point", "pt")):
            resolved_unit = "point"
        elif unit in {"x", "×"}:
            resolved_unit = "x"
        else:
            resolved_unit = unit
    return {"raw": num, "normalized": normalized, "unit": resolved_unit}


def _local_metric(text: str, start: int, end: int) -> str | None:
    before = re.split(r"[,;:]", text[max(0, start - 24) : start])[-1]
    after = re.split(r"[,;:.]", text[end : end + 32])[0]
    local = before + text[start:end] + after
    match = METRIC_RE.search(local)
    return match.group(0).lower() if match else None


def _number_value(text: str, match: re.Match[str]) -> dict[str, Any]:
    value = _parse_number(match.group(1), match.group(2))
    value.update(
        {
            "metric": _local_metric(text, match.start(1), match.end()),
            "direction": "unknown",
            "aggregation": "unspecified",
        }
    )
    return value


def _source_record(
    path: Path, *, kind: str, sha256: str | None = None
) -> dict[str, Any]:
    return {
        "path": str(path),
        "kind": kind,
        "sha256": sha256 if sha256 is not None else _sha256_file(path),
    }


def _claims_from_sentence(
    *,
    sentence: str,
    source: Path,
    line_no: int | None,
    section: str,
    anchor: str,
    extractor: str,
    confidence: str,
) -> list[dict[str, Any]]:
    location: dict[str, Any] = {"file": str(source), "section": section}
    if line_no is not None:
        location["line"] = line_no
    claims: list[dict[str, Any]] = []
    for match in NUMBER_RE.finditer(sentence):
        num, unit = match.group(1), match.group(2)
        if not unit and "." not in num and not METRIC_RE.search(sentence):
            continue
        claims.append(
            {
                "type": "number",
                "text_span": sentence[:400],
                "location": dict(location),
                "value": _number_value(sentence, match),
                "evidence_anchor": anchor,
                "extractor": extractor,
                "confidence": confidence,
            }
        )
    for cite_match in CITE_RE.finditer(sentence):
        refs = [key.strip() for key in cite_match.group(1).split(",") if key.strip()]
        claims.append(
            {
                "type": "citation",
                "text_span": sentence[:400],
                "location": dict(location),
                "refs": refs,
                "evidence_anchor": anchor,
                "extractor": extractor,
                "confidence": confidence,
            }
        )
    if SCOPE_RE.search(sentence):
        claims.append(
            {
                "type": "scope",
                "text_span": sentence[:400],
                "location": dict(location),
                "evidence_anchor": anchor,
                "extractor": extractor,
                "confidence": confidence,
            }
        )
    if any(phrase in sentence.lower() for phrase in PIPELINE_ARTIFACT_PHRASES):
        claims.append(
            {
                "type": "presentation",
                "text_span": sentence[:400],
                "location": dict(location),
                "evidence_anchor": anchor,
                "extractor": extractor,
                "confidence": confidence,
            }
        )
    return claims


def extract_claims_from_latex(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract checkable span claims from one LaTeX source file."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    anchor = _sha256_text(text)
    lines = text.splitlines()
    label = _section_labeler(lines)
    tab_lines = _tabular_lines(lines)
    claims: list[dict[str, Any]] = []

    for line_no, line in enumerate(lines, start=1):
        if line_no not in tab_lines or line.strip().startswith("%"):
            continue
        section = label(line_no)
        for match in NUMBER_RE.finditer(line):
            claims.append(
                {
                    "type": "table_cell",
                    "text_span": line.strip()[:300],
                    "location": {
                        "file": str(source),
                        "line": line_no,
                        "section": section,
                    },
                    "value": _number_value(line, match),
                    "evidence_anchor": anchor,
                    "extractor": "latex_table_regex",
                    "confidence": "medium",
                }
            )

    for block, start_line in _iter_prose_blocks(lines, tab_lines=tab_lines):
        section = label(start_line)
        for sentence in _split_sentences(block):
            claims.extend(
                _claims_from_sentence(
                    sentence=sentence,
                    source=source,
                    line_no=start_line,
                    section=section,
                    anchor=anchor,
                    extractor="latex_regex",
                    confidence="high",
                )
            )

    for paragraph, start, end in _iter_paragraphs(lines):
        section = label(start)
        if CAPTION_RE.search(paragraph):
            claims.append(
                {
                    "type": "caption",
                    "text_span": paragraph[paragraph.find("\\caption") :][:400],
                    "location": {
                        "file": str(source),
                        "line": start,
                        "section": section,
                    },
                    "evidence_anchor": anchor,
                    "extractor": "latex_regex",
                    "confidence": "medium",
                }
            )

    return claims, _source_record(source, kind="latex", sha256=anchor)


def extract_claims_from_text(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Extract lower-confidence claims from plain text, such as PDF extraction."""

    source = Path(path)
    text = source.read_text(encoding="utf-8", errors="replace")
    anchor = _sha256_text(text)
    claims: list[dict[str, Any]] = []
    for paragraph in re.split(r"\n\s*\n", text):
        for sentence in _split_sentences(paragraph.replace("\n", " ")):
            claims.extend(
                _claims_from_sentence(
                    sentence=sentence,
                    source=source,
                    line_no=None,
                    section="unknown",
                    anchor=anchor,
                    extractor="text_regex",
                    confidence="low",
                )
            )
    return claims, _source_record(source, kind="text", sha256=anchor)


def build_integrity_ledger(
    *,
    paper_id: str,
    latex_paths: Sequence[str | Path] = (),
    text_paths: Sequence[str | Path] = (),
    observability_level: int = 1,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a span-anchored claim ledger from manuscript sources."""

    if not latex_paths and not text_paths:
        raise ValueError("provide at least one LaTeX or text source")
    if observability_level < 0:
        raise ValueError("observability_level must be non-negative")

    claims: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in latex_paths:
        extracted, source = extract_claims_from_latex(path)
        claims.extend(extracted)
        sources.append(source)
    for path in text_paths:
        extracted, source = extract_claims_from_text(path)
        claims.extend(extracted)
        sources.append(source)

    indexed = [
        {"claim_id": f"C{idx:03d}", **claim} for idx, claim in enumerate(claims, 1)
    ]
    return IntegrityLedger(
        paper_id=str(paper_id),
        observability_level=int(observability_level),
        claims=indexed,
        source_files=sources,
        generated_at=generated_at or _now_iso(),
    ).to_dict()


def _relative_delta(old: float, new: float) -> float:
    return (new - old) / old * 100.0 if old else float("inf")


def check_numeric_consistency(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Run deterministic numeric consistency checks over a ledger."""

    claims = list(ledger.get("claims") or [])
    findings = _check_delta_consistency(claims)
    findings.extend(_check_headline_numbers_in_tables(claims, start=len(findings)))
    return findings


def _check_delta_consistency(claims: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for claim in claims:
        raw = str(claim.get("text_span") or "")
        span = raw.replace("\\%", "%").replace("\\,", " ")
        location = claim.get("location") or {}
        for pattern_index, pattern in enumerate(DELTA_PATTERNS):
            for match in pattern.finditer(span):
                if pattern_index == 0:
                    old_s, new_s, stated_s, rel_word, direction = match.groups()
                else:
                    stated_s, rel_word, direction, old_s, new_s = match.groups()
                key = (
                    location.get("file"),
                    location.get("line"),
                    old_s,
                    new_s,
                    stated_s,
                )
                if key in seen:
                    continue
                seen.add(key)
                old, new, stated = float(old_s), float(new_s), float(stated_s)
                relative = _relative_delta(old, new)
                absolute = new - old
                decrease = bool(
                    re.search(r"reduc|drop|decreas|lower|fewer", direction or "", re.I)
                )
                rel_cmp = abs(relative) if decrease else relative
                abs_cmp = abs(absolute) if decrease else absolute
                claims_relative = bool(rel_word)
                if claims_relative:
                    ok = abs(stated - rel_cmp) <= DELTA_TOLERANCE
                else:
                    ok = (
                        abs(stated - rel_cmp) <= DELTA_TOLERANCE
                        or abs(stated - abs(abs_cmp)) <= DELTA_TOLERANCE
                    )
                if ok:
                    continue
                finding_id = f"NUM{len(findings) + 1:03d}"
                findings.append(
                    {
                        "finding_id": finding_id,
                        "skill": "consistency-audit",
                        "pattern_id": "HP-DELTA-ERROR",
                        "title": "Stated change contradicts its operands",
                        "description": (
                            f"Text states a {stated:g}% "
                            f"{'relative ' if claims_relative else ''}change, but "
                            f"{old:g}->{new:g} is {relative:.1f}% relative "
                            f"({absolute:+.1f} absolute points)."
                        ),
                        "severity": "major",
                        "observability_level_required": 0,
                        "evidence": [
                            {
                                "claim_id": claim.get("claim_id"),
                                "span": raw,
                                "location": location,
                                "artifact_hash": claim.get("evidence_anchor", ""),
                            }
                        ],
                        "verdict_local": "fail",
                        "reviewer": {"deterministic": True},
                        "false_positive_risk": "low",
                        "recommended_reviewer_action": (
                            "Reconcile the stated delta with the reported operands and "
                            "the relative-vs-absolute convention."
                        ),
                    }
                )
    return findings


def _check_headline_numbers_in_tables(
    claims: Sequence[dict[str, Any]], *, start: int = 0
) -> list[dict[str, Any]]:
    table_values: set[float] = set()
    for claim in claims:
        if claim.get("type") != "table_cell":
            continue
        value = (claim.get("value") or {}).get("normalized")
        if isinstance(value, (int, float)):
            table_values.add(round(float(value), 1))
    if not table_values:
        return []

    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for claim in claims:
        if claim.get("type") != "number":
            continue
        text_span = str(claim.get("text_span") or "")
        if re.search(
            r"\b(from\b.+\bto\b|improv\w*|gain|increase|boost|reduc\w*|"
            r"decreas\w*|drop|relative|absolute)\b",
            text_span,
            re.IGNORECASE,
        ):
            continue
        location = claim.get("location") or {}
        section = str(location.get("section") or "").lower()
        value = claim.get("value") or {}
        normalized = value.get("normalized")
        if value.get("unit") != "%" or not value.get("metric"):
            continue
        if not isinstance(normalized, (int, float)):
            continue
        if not any(key in section for key in ("abstract", "intro", "conclusion")):
            continue
        rounded = round(float(normalized), 1)
        if rounded in table_values:
            continue
        key = (section, rounded)
        if key in seen:
            continue
        seen.add(key)
        finding_id = f"NUM{start + len(findings) + 1:03d}"
        findings.append(
            {
                "finding_id": finding_id,
                "skill": "consistency-audit",
                "pattern_id": "HP-NUM-INFLATE",
                "title": "Headline number not found in extracted result tables",
                "description": (
                    f"The {section} cites {rounded:g}% {value.get('metric')}, but no "
                    "extracted table cell reports that value. It may be a different "
                    "setting, a rounding artifact, or an inflated headline."
                ),
                "severity": "minor",
                "observability_level_required": 0,
                "evidence": [
                    {
                        "claim_id": claim.get("claim_id"),
                        "span": claim.get("text_span", ""),
                        "location": location,
                        "artifact_hash": claim.get("evidence_anchor", ""),
                    }
                ],
                "verdict_local": "warn",
                "reviewer": {"deterministic": True},
                "false_positive_risk": "high",
                "recommended_reviewer_action": (
                    "Locate the source table/row for the headline number and confirm "
                    "the setting it refers to."
                ),
            }
        )
    return findings


def check_statistical_consistency(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Run conservative deterministic statistical checks."""

    claims = list(ledger.get("claims") or [])
    return _check_grim_consistency(claims)


def _decimals(num: str) -> int:
    return len(num.split(".", 1)[1]) if "." in num else 0


def _grim_vacuous(n: int, decimals: int) -> bool:
    return (100.0 / n) <= (10 ** (-decimals)) + 1e-12


def _grim_achievable(percent: float, n: int, decimals: int) -> bool:
    half_ulp = 0.5 * (10 ** (-decimals))
    target = percent / 100.0 * n
    low = max(0, math.floor(target) - 1)
    high = min(n, math.ceil(target) + 1)
    return any(
        abs(100.0 * k / n - percent) <= half_ulp + 1e-9 for k in range(low, high + 1)
    )


def _grim_nearest(percent: float, n: int, decimals: int) -> float | None:
    target = percent / 100.0 * n
    best: float | None = None
    for k in (math.floor(target), round(target), math.ceil(target)):
        if 0 <= k <= n:
            candidate = round(100.0 * k / n, decimals)
            if best is None or abs(candidate - percent) < abs(best - percent):
                best = candidate
    return best


def _check_grim_consistency(claims: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for claim in claims:
        if claim.get("type") not in {"number", "table_cell"}:
            continue
        raw = str(claim.get("text_span") or "")
        span = raw.replace("\\%", "%").replace("\\,", " ").replace("~", " ")
        if not GRIM_METRIC_RE.search(span):
            continue
        if GRIM_EXCLUDE_RE.search(span):
            continue
        denominators = sorted(
            {
                int(group)
                for match in GRIM_N_RE.finditer(span)
                for group in match.groups()
                if group
            }
        )
        if len(denominators) != 1:
            continue
        n = denominators[0]
        if n <= 0:
            continue
        for percent_match in PERCENT_RE.finditer(span):
            percent_s = percent_match.group(1)
            percent = float(percent_s)
            if not (0.0 < percent <= 100.0):
                continue
            decimals = _decimals(percent_s)
            if _grim_vacuous(n, decimals) or _grim_achievable(percent, n, decimals):
                continue
            location = claim.get("location") or {}
            key = (location.get("file"), location.get("line"), percent_s, n)
            if key in seen:
                continue
            seen.add(key)
            nearest = _grim_nearest(percent, n, decimals)
            section = str(location.get("section") or "").lower()
            headline = any(
                key in section for key in ("abstract", "intro", "conclusion")
            )
            findings.append(
                {
                    "finding_id": f"STAT{len(findings) + 1:03d}",
                    "skill": "consistency-audit",
                    "pattern_id": "HP-GRANULARITY-IMPOSSIBLE",
                    "title": "Reported proportion is not achievable for the stated N",
                    "description": (
                        f"{percent:g}% over N={n} integer items is not round(k/{n}) "
                        f"at {decimals} decimal place(s) for any integer k"
                        + (
                            f" (nearest achievable {nearest:g}%)."
                            if nearest is not None
                            else "."
                        )
                    ),
                    "severity": "major" if headline else "minor",
                    "observability_level_required": 0,
                    "evidence": [
                        {
                            "claim_id": claim.get("claim_id"),
                            "span": raw,
                            "location": location,
                            "artifact_hash": claim.get("evidence_anchor", ""),
                        }
                    ],
                    "verdict_local": "fail" if headline else "warn",
                    "reviewer": {"deterministic": True, "method": "GRIM"},
                    "false_positive_risk": "low",
                    "recommended_reviewer_action": (
                        "Reconcile the value, display precision, and exact denominator N."
                    ),
                }
            )
    return findings


def check_presentation_signals(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    """Run deterministic presentation/surface-signal checks."""

    claims = list(ledger.get("claims") or [])
    findings = _check_duplicate_tables(claims)
    findings.extend(_check_pipeline_artifacts(claims, start=len(findings)))
    return findings


def _table_signatures(claims: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for claim in claims:
        if claim.get("type") != "table_cell":
            continue
        location = claim.get("location") or {}
        section = str(location.get("section") or "")
        if not re.match(r"table:\d+", section):
            continue
        table_key = f"{location.get('file', '')}#{section}"
        value = (claim.get("value") or {}).get("normalized")
        if not isinstance(value, (int, float)):
            continue
        tables.setdefault(
            table_key,
            {
                "values": [],
                "claim_id": claim.get("claim_id"),
                "span": claim.get("text_span", ""),
                "anchor": claim.get("evidence_anchor", ""),
                "location": location,
                "label": section,
            },
        )
        tables[table_key]["values"].append(round(float(value), 4))
    return tables


def _check_duplicate_tables(claims: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    tables = _table_signatures(claims)
    keys = sorted(tables)
    findings: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for idx, left_key in enumerate(keys):
        for right_key in keys[idx + 1 :]:
            left = tables[left_key]
            right = tables[right_key]
            if len(left["values"]) < 2 or len(right["values"]) < 2:
                continue
            if left["values"] != right["values"]:
                continue
            pair = (left_key, right_key)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            findings.append(
                {
                    "finding_id": f"PRES{len(findings) + 1:03d}",
                    "skill": "presentation-signals",
                    "pattern_id": "HP-DUP-TABLE",
                    "title": "Two tables have identical numeric content",
                    "description": (
                        f"{left.get('label', left_key)} and {right.get('label', right_key)} "
                        "contain the same ordered cell "
                        f"values ({left['values']})."
                    ),
                    "severity": "minor",
                    "observability_level_required": 0,
                    "evidence": [
                        {
                            "claim_id": left["claim_id"],
                            "span": left["span"],
                            "location": left["location"],
                            "artifact_hash": left["anchor"],
                        },
                        {
                            "claim_id": right["claim_id"],
                            "span": right["span"],
                            "location": right["location"],
                            "artifact_hash": right["anchor"],
                        },
                    ],
                    "verdict_local": "warn",
                    "reviewer": {"deterministic": True},
                    "false_positive_risk": "high",
                    "recommended_reviewer_action": (
                        "Check whether the two tables are meant to differ; if not, "
                        "remove or explain the duplicate."
                    ),
                }
            )
    return findings


def _anchorable(span: Any) -> bool:
    normalized = _norm_ws(span)
    if not any(char.isalnum() for char in normalized):
        return False
    return len(normalized) >= 12 or len(normalized.split()) >= 3


def _anchor_window(text: str, start: int, end: int, *, pad: int = 28) -> str:
    window = text[max(0, start - pad) : min(len(text), end + pad)].strip()
    if _anchorable(window):
        return window
    return text.strip()


def _check_pipeline_artifacts(
    claims: Sequence[dict[str, Any]], *, start: int = 0
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for claim in claims:
        span_text = claim.get("text_span")
        if not isinstance(span_text, str) or not span_text:
            continue
        location = claim.get("location") or {}
        for phrase in PIPELINE_ARTIFACT_PHRASES:
            for match in re.finditer(re.escape(phrase), span_text, re.IGNORECASE):
                hit = match.group(0)
                key = (
                    location.get("file"),
                    location.get("section"),
                    location.get("line"),
                    _norm_ws(span_text),
                    phrase,
                )
                if key in seen:
                    continue
                seen.add(key)
                findings.append(
                    {
                        "finding_id": f"PRES{start + len(findings) + 1:03d}",
                        "skill": "presentation-signals",
                        "pattern_id": "HP-PIPELINE-ARTIFACT",
                        "title": "Leftover pipeline/assistant string in finished text",
                        "description": (
                            f'The exact phrase "{hit}" appears in '
                            f"{location.get('section', '?')}. This is a checkable "
                            "leftover string, not an authorship verdict."
                        ),
                        "severity": "minor",
                        "observability_level_required": 0,
                        "evidence": [
                            {
                                "claim_id": claim.get("claim_id"),
                                "span": _anchor_window(
                                    span_text, match.start(), match.end()
                                ),
                                "location": location,
                                "artifact_hash": claim.get("evidence_anchor", ""),
                            }
                        ],
                        "verdict_local": "warn",
                        "reviewer": {"deterministic": True},
                        "false_positive_risk": "low",
                        "recommended_reviewer_action": (
                            "Confirm the phrase is a genuine leftover rather than a "
                            "deliberate quotation or object of study."
                        ),
                    }
                )
    return findings


def _cap_severity(severity: str, cap: str) -> str:
    return severity if SEVERITY_ORDER[severity] <= SEVERITY_ORDER[cap] else cap


def _anchored(finding: dict[str, Any], ledger_map: dict[str, str]) -> bool:
    for evidence in finding.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        claim_id = evidence.get("claim_id")
        span = _norm_ws(evidence.get("span"))
        if not claim_id or not _anchorable(span):
            continue
        base = ledger_map.get(str(claim_id))
        if base is not None and span in _norm_ws(base):
            return True
    return False


def adjudicate_findings(
    findings: Sequence[dict[str, Any]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    """Apply deterministic gates and compute the integrity verdict."""

    ledger_claims = [
        claim for claim in ledger.get("claims") or [] if isinstance(claim, dict)
    ]
    ledger_map = {
        str(claim.get("claim_id")): str(claim.get("text_span") or "")
        for claim in ledger_claims
        if claim.get("claim_id")
    }
    try:
        run_level = int(ledger.get("observability_level", 0))
    except (TypeError, ValueError):
        run_level = 0

    adjudicated: list[dict[str, Any]] = []
    for raw in findings:
        finding = dict(raw)
        severity = str(finding.get("severity") or "info").lower()
        if severity not in SEVERITY_ORDER:
            severity = "info"
        reasons: list[str] = []

        if severity != "info" and not _anchored(finding, ledger_map):
            severity = "info"
            reasons.append("anchor-gate-demoted")

        try:
            required_level = int(finding.get("observability_level_required", 0))
        except (TypeError, ValueError):
            required_level = run_level + 1
        if required_level > run_level and severity != "info":
            severity = "info"
            reasons.append("observability-gate-demoted")

        fp_risk = str(finding.get("false_positive_risk") or "medium").lower()
        if fp_risk == "high":
            capped = _cap_severity(severity, "minor")
            if capped != severity:
                reasons.append("false-positive-risk-cap-high")
            severity = capped
        elif fp_risk == "medium":
            capped = _cap_severity(severity, "major")
            if capped != severity:
                reasons.append("false-positive-risk-cap-medium")
            severity = capped

        pattern_id = str(finding.get("pattern_id") or "")
        skill = str(finding.get("skill") or "")
        if skill == "presentation-signals" or pattern_id.startswith("HP-PIPELINE"):
            capped = _cap_severity(severity, "minor")
            if capped != severity:
                reasons.append("surface-signal-cap")
            severity = capped

        if (
            pattern_id.startswith(("AIS-", "ADV-"))
            or skill in {"ai-style-impressions", "adversarial-case-builder"}
            or finding.get("requires_external_check") is True
        ):
            if severity != "info":
                reasons.append("zero-weight-or-external-check")
            severity = "info"

        finding["_severity_final"] = severity
        finding["_verdict_weight"] = 0 if severity == "info" else 1
        finding["_adjudication"] = reasons or ["passed"]
        adjudicated.append(finding)

    max_severity = max(
        (SEVERITY_ORDER[item["_severity_final"]] for item in adjudicated),
        default=0,
    )
    if max_severity >= SEVERITY_ORDER["critical"]:
        verdict = "HARD_FLAGS"
    elif max_severity >= SEVERITY_ORDER["minor"]:
        verdict = "SOFT_FLAGS"
    else:
        verdict = "CLEAN_GIVEN_EVIDENCE"

    counts_by_severity = {
        name: sum(1 for item in adjudicated if item["_severity_final"] == name)
        for name in SEVERITY_ORDER
    }
    counts_by_pattern: dict[str, int] = {}
    for item in adjudicated:
        pattern = str(item.get("pattern_id") or "UNKNOWN")
        counts_by_pattern[pattern] = counts_by_pattern.get(pattern, 0) + 1

    return {
        "report_version": REPORT_VERSION,
        "adjudicator": ADJUDICATOR_ID,
        "paper_id": ledger.get("paper_id"),
        "generated_at": _now_iso(),
        "observability_level": run_level,
        "overall_verdict": verdict,
        "counts": {
            "claims": len(ledger_claims),
            "findings": len(adjudicated),
            "by_severity": counts_by_severity,
            "by_pattern": dict(sorted(counts_by_pattern.items())),
        },
        "findings": adjudicated,
        "limitations": _report_limitations(ledger),
    }


def _report_limitations(ledger: dict[str, Any]) -> list[str]:
    level = ledger.get("observability_level", 0)
    limitations = [
        "Verdict means integrity under available evidence, not an authorship or misconduct decision."
    ]
    limitations.append(
        "Granularity checks only compare denominators and percentages that appear in the same extracted span."
    )
    if level < 2:
        limitations.append(
            "Code/result-level fabrication checks are observability-limited without result artifacts."
        )
    if not ledger.get("claims"):
        limitations.append(
            "No checkable claims were extracted from the provided sources."
        )
    return limitations


def render_integrity_report_markdown(report: dict[str, Any]) -> str:
    """Render a compact reviewer-facing integrity report."""

    lines = [
        "# Integrity Forensics Report",
        "",
        f"- Paper ID: {report.get('paper_id')}",
        f"- Verdict: {report.get('overall_verdict')}",
        f"- Observability level: L{report.get('observability_level')}",
        f"- Adjudicator: {report.get('adjudicator')}",
        "",
        "## Summary",
    ]
    counts = report.get("counts") or {}
    by_severity = counts.get("by_severity") or {}
    lines.append(f"- Claims extracted: {counts.get('claims', 0)}")
    lines.append(f"- Findings: {counts.get('findings', 0)}")
    lines.append(
        "- Severity counts: "
        + ", ".join(f"{key}={by_severity.get(key, 0)}" for key in SEVERITY_ORDER)
    )

    findings = [
        item
        for item in report.get("findings") or []
        if item.get("_severity_final") != "info"
    ]
    lines.extend(["", "## Findings"])
    if not findings:
        lines.append("- No verdict-bearing findings under available evidence.")
    else:
        for item in findings[:20]:
            evidence = item.get("evidence") or []
            first_evidence = (
                evidence[0] if evidence and isinstance(evidence[0], dict) else {}
            )
            location = first_evidence.get("location") or {}
            where = location.get("file", "?")
            if location.get("line"):
                where = f"{where}:{location.get('line')}"
            lines.append(
                f"- [{item.get('_severity_final')}] {item.get('pattern_id')}: "
                f"{item.get('title')} ({where})"
            )
            description = str(item.get("description") or "").strip()
            if description:
                lines.append(f"  {description}")

    limitations = report.get("limitations") or []
    lines.extend(["", "## Limitations"])
    for limitation in limitations:
        lines.append(f"- {limitation}")
    return "\n".join(lines) + "\n"


def run_integrity_forensics(
    *,
    paper_id: str,
    latex_paths: Sequence[str | Path] = (),
    text_paths: Sequence[str | Path] = (),
    output_dir: str | Path | None = None,
    observability_level: int = 1,
) -> IntegrityRunResult:
    """Run the deterministic ledger/check/adjudication pipeline."""

    ledger = build_integrity_ledger(
        paper_id=paper_id,
        latex_paths=latex_paths,
        text_paths=text_paths,
        observability_level=observability_level,
    )
    findings: list[dict[str, Any]] = []
    findings.extend(check_numeric_consistency(ledger))
    findings.extend(check_statistical_consistency(ledger))
    findings.extend(check_presentation_signals(ledger))
    report = adjudicate_findings(findings, ledger)

    files: dict[str, str] = {}
    output_path: Path | None = None
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "claims": output_path / "claims.json",
            "findings": output_path / "integrity-findings.json",
            "report": output_path / "report.json",
            "markdown": output_path / "REPORT.md",
        }
        artifacts["claims"].write_text(
            json.dumps(ledger, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts["findings"].write_text(
            json.dumps(findings, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts["report"].write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        artifacts["markdown"].write_text(
            render_integrity_report_markdown(report),
            encoding="utf-8",
        )
        files = {key: str(path) for key, path in artifacts.items()}

    return IntegrityRunResult(
        ledger=ledger,
        findings=findings,
        report=report,
        output_dir=str(output_path) if output_path is not None else None,
        files=files,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deterministic integrity forensics."
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--latex", nargs="*", default=[])
    parser.add_argument("--text", nargs="*", default=[])
    parser.add_argument("--observability-level", type=int, default=1)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_cli_parser()
    args = parser.parse_args(argv)
    result = run_integrity_forensics(
        paper_id=args.paper_id,
        latex_paths=args.latex,
        text_paths=args.text,
        output_dir=args.out_dir,
        observability_level=args.observability_level,
    )
    print(
        "integrity forensics: "
        f"{result.report.get('overall_verdict')} "
        f"claims={result.report.get('counts', {}).get('claims')} "
        f"findings={result.report.get('counts', {}).get('findings')} "
        f"-> {result.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
