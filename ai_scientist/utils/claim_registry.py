"""Claim registry — link every manuscript assertion to its exploration node.

Background
----------
Part of the ARA (Agent-Native Research Artifact) push (see `ara_artifact.py`).
A PDF loses the mapping between "we obtained 12.4% accuracy" and *which tree
search node* produced that number. This module restores that mapping.

Contract
--------
The writeup prompt is given a LaTeX macro:

    \\newcommand{\\claimref}[2][]{}

It renders as nothing in the PDF, but leaves a machine-readable marker of the
form ``\\claimref[key=value,...]{node_id}`` (or ``\\claimref{node_id}``) inside
the .tex source. This module scans the final .tex, resolves each marker
against the exploration graph, and writes one JSON file per claim under
``<ara_dir>/claims/``.

We do not fail the pipeline if no markers are found — an unmarked PDF is still
a valid ARA, just with `claims_count == 0` and a note in `manifest.missing`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

CLAIM_MACRO_NAME = "claimref"
CLAIM_MACRO_LATEX = (
    r"% ARA claim marker — no visual output; consumed by claim_registry.py"
    "\n"
    r"\providecommand{\claimref}[2][]{}"
    "\n"
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Matches ``\claimref{node_id}`` or ``\claimref[key=val, key2=val2]{node_id}``.
# We tolerate whitespace and multiple spaces inside the optional bracket.
_CLAIM_RE = re.compile(
    r"\\claimref\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{\s*(?P<node>[^{}\s,]+)\s*\}",
    re.MULTILINE,
)


@dataclass
class Claim:
    claim_id: str
    node_id: str
    tex_file: str
    line: int
    context: str
    options: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_opts(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    out: dict[str, str] = {}
    # Allow either `key=value` or bare tokens (recorded under `_flags`).
    flags: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, _, v = chunk.partition("=")
            k = k.strip()
            v = v.strip().strip("'\"")
            if k:
                out[k] = v
        else:
            flags.append(chunk)
    if flags:
        out["_flags"] = ",".join(flags)
    return out


def _line_of(offset: int, text: str) -> int:
    return text.count("\n", 0, offset) + 1


def _context_around(text: str, offset: int, span: int = 120) -> str:
    start = max(0, offset - span)
    end = min(len(text), offset + span)
    excerpt = text[start:end].replace("\n", " ")
    return re.sub(r"\s+", " ", excerpt).strip()


def scan_tex_for_claims(tex_path: str | Path, *, tex_root: str | Path | None = None) -> list[Claim]:
    r"""Extract every ``\claimref{...}`` marker from a .tex file."""
    tex_path = Path(tex_path)
    if not tex_path.exists():
        return []
    try:
        text = tex_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("claim_registry: cannot read %s: %s", tex_path, exc)
        return []

    rel: str
    if tex_root is not None:
        try:
            rel = str(tex_path.relative_to(Path(tex_root)))
        except ValueError:
            rel = str(tex_path)
    else:
        rel = str(tex_path)

    claims: list[Claim] = []
    seen: set[tuple[str, int]] = set()
    for match in _CLAIM_RE.finditer(text):
        node_id = match.group("node").strip()
        if not node_id:
            continue
        line = _line_of(match.start(), text)
        key = (node_id, line)
        if key in seen:
            continue
        seen.add(key)
        options = _parse_opts(match.group("opts"))
        claim_id = f"{Path(rel).stem}_{line}_{node_id}"
        claims.append(
            Claim(
                claim_id=claim_id,
                node_id=node_id,
                tex_file=rel,
                line=line,
                context=_context_around(text, match.start()),
                options=options,
            )
        )
    return claims


def _load_exploration_index(ara_dir: Path) -> dict[str, dict[str, Any]]:
    graph_path = ara_dir / "exploration_graph.json"
    if not graph_path.exists():
        return {}
    try:
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    index: dict[str, dict[str, Any]] = {}
    for node in payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if isinstance(nid, str) and nid:
            index[nid] = node
    return index


def write_claims_into_ara(
    *,
    ara_dir: str | Path,
    tex_files: Iterable[str | Path],
) -> dict[str, Any]:
    """Scan tex_files, resolve node references, write per-claim JSON.

    Returns a summary dict with counts and lists of unresolved node ids so
    callers can report / persist them.
    """
    ara_dir = Path(ara_dir)
    claims_root = ara_dir / "claims"
    claims_root.mkdir(parents=True, exist_ok=True)

    index = _load_exploration_index(ara_dir)

    all_claims: list[Claim] = []
    for tex in tex_files:
        all_claims.extend(scan_tex_for_claims(tex, tex_root=Path(tex).parent))

    unresolved: list[str] = []
    written: list[str] = []
    for claim in all_claims:
        node_meta = index.get(claim.node_id)
        payload = {
            **claim.to_dict(),
            "resolved": node_meta is not None,
            "node": node_meta,
            "recorded_at": _now_iso(),
        }
        target = claims_root / f"{claim.claim_id}.json"
        try:
            target.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("claim_registry: cannot write %s: %s", target, exc)
            continue
        written.append(str(target.relative_to(ara_dir)))
        if node_meta is None:
            unresolved.append(claim.node_id)

    summary = {
        "claim_count": len(all_claims),
        "resolved_count": len(all_claims) - len(unresolved),
        "unresolved_node_ids": sorted(set(unresolved)),
        "files_written": written,
        "generated_at": _now_iso(),
    }
    summary_path = claims_root / "_index.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def render_claim_prompt_snippet(*, max_nodes: int = 20) -> str:
    """Prompt fragment we can splice into writeup prompts.

    Kept intentionally short — the goal is to teach the LLM the *syntax* and
    the *when*, not to over-scaffold. The bitter lesson warning in the file
    header applies here too.
    """
    return (
        "When you cite an experimental result (a number, a table cell, a "
        "figure claim), append `\\claimref{<node_id>}` immediately after the "
        "sentence. `<node_id>` is the tree-search node that produced the "
        f"evidence (see the exploration_graph). Up to {max_nodes} claims per "
        "section is plenty. This macro is invisible in the PDF but lets a "
        "downstream agent re-execute the corresponding node."
    )
