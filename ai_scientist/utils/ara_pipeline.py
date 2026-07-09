"""High-level ARA glue for ``run_project.py`` and other pipeline entrypoints.

Rationale
---------
Before this module existed, ``run_project.py`` had ARA logic in three places:

1. CLI-time seed staging (about 30 lines).
2. Per-idea ``export_ara`` call with ad-hoc provenance decoding.
3. Optional re-execution wired via inline ``if reexec_enabled(): ...``.

Each of those referenced env-var names as raw strings and duplicated
Path/JSON handling. This module concentrates the surface so the pipeline
becomes ~5 lines:

    stage_result = stage_seed_from_cli(args, project_dir=Path(project_dir))
    ...
    ara_export = finalize_ara_for_idea(
        exp_dir=exp_dir, project_dir=project_dir, idea=idea, ...
    )

The env vars are still the transport mechanism (they cross subprocess
boundaries for free) but their names are now defined here in one place.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_scientist.utils.ara_artifact import (
    ARAExportResult,
    ara_dir_for_idea,
    export_ara,
    update_manifest_claim_count,
)
from ai_scientist.utils.ara_reexec import reexec_ara, reexec_enabled
from ai_scientist.utils.ara_seed import (
    SEED_ENV_VAR,
    build_seed_manifest_from_ara_node,
    resolve_seed_manifest_from_source,
    stage_seed_manifest,
)
from ai_scientist.protocol.llm_trace import (
    ENV_ACTIVE_ROOT as LLM_TRACE_ROOT_ENV,
    ENV_STAGE as LLM_TRACE_STAGE_ENV,
)
from ai_scientist.utils.claim_coverage import (
    ClaimCoverageReport,
    evaluate_claim_coverage,
    write_coverage_into_ara,
)
from ai_scientist.utils.claim_registry import write_claims_into_ara

logger = logging.getLogger(__name__)

# Env var used to carry the *parent's* provenance dict across subprocess
# boundaries. Kept next to SEED_ENV_VAR so the two never drift.
SEED_PROVENANCE_ENV_VAR = "AI_SCIENTIST_ARA_SEED_PROVENANCE"


@dataclass
class SeedStageResult:
    """What happened when we tried to stage a seed at CLI time."""

    seed_used: bool
    seed_path: Path | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ARAFinalizeResult:
    """What happened when we tried to finalize the per-idea ARA."""

    export: ARAExportResult | None = None
    claim_summary: dict[str, Any] | None = None
    claim_coverage: ClaimCoverageReport | None = None
    reexec_summary: dict[str, Any] | None = None
    error: str | None = None


def stage_seed_from_cli(
    *,
    seed_from_ara: str | None,
    seed_node_id: str | None,
    project_dir: Path | str,
) -> SeedStageResult:
    """Turn ``--seed-from-ara`` / ``--seed-node-id`` into a staged manifest.

    Sets ``SEED_ENV_VAR`` and ``SEED_PROVENANCE_ENV_VAR`` so downstream
    processes (BFTS workers, ``process_single_idea``) pick them up. Callers
    handle the ``error`` field — no exceptions escape.
    """
    if not seed_from_ara:
        return SeedStageResult(seed_used=False)

    try:
        if seed_node_id:
            manifest = build_seed_manifest_from_ara_node(
                ara_root=seed_from_ara, node_id=seed_node_id
            )
        else:
            manifest = resolve_seed_manifest_from_source(seed_from_ara)
    except (FileNotFoundError, ValueError) as exc:
        return SeedStageResult(seed_used=False, error=str(exc))

    seed_path = stage_seed_manifest(
        manifest, workspace_dir=Path(project_dir) / ".ara_seed"
    )
    os.environ[SEED_ENV_VAR] = str(seed_path)
    provenance = manifest.get("provenance") or {}
    os.environ[SEED_PROVENANCE_ENV_VAR] = json.dumps(provenance)
    return SeedStageResult(
        seed_used=True, seed_path=seed_path, provenance=provenance
    )


def _read_provenance_env() -> dict[str, Any] | None:
    """Fetch the provenance dict a parent staged for us. None when absent."""
    raw = os.environ.get(SEED_PROVENANCE_ENV_VAR)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if (isinstance(parsed, dict) and parsed) else None


def activate_llm_tracing(
    *,
    project_dir: str | Path,
    idea: dict[str, Any] | None = None,
    exp_dir: str | Path | None = None,
    timestamp: str | None = None,
    stage: str | None = None,
) -> Path | None:
    """Point the LLM tracer at the ARA directory this idea will write into.

    Rationale: LLM calls happen BEFORE ``export_ara`` runs — if we only turn
    tracing on after export, we've missed every prompt that generated the
    code we're exporting. So we resolve the ARA directory eagerly (same math
    ``export_ara`` uses at line 589), pre-create it, and set the env vars
    that ``ai_scientist.protocol.llm_trace.record_llm_call`` reads.

    Passing ``stage`` sets a default group label; individual stages can still
    override it by re-setting ``AI_SCIENTIST_LLM_STAGE`` before their own
    calls. Returns the ARA root that was activated, or None if we couldn't
    resolve one (e.g. the caller supplied no idea + exp_dir hint).
    """
    idea = idea or {}
    exp_path = Path(exp_dir).expanduser().resolve() if exp_dir else None
    name = str(idea.get("Name") or idea.get("name") or (exp_path.name if exp_path else "")).strip()
    if not name:
        return None

    ts = timestamp
    if ts is None and exp_path is not None:
        # Same regex ``export_ara`` uses when extracting the timestamp segment.
        import re as _re
        m = _re.match(r"^(\d{4}-\d{2}-\d{2}[_T-]?\d{2}[:_-]?\d{2}(?:[:_-]?\d{2})?)", exp_path.name)
        ts = m.group(1) if m else None

    try:
        ara_dir = ara_dir_for_idea(Path(project_dir).expanduser().resolve(),
                                   name, timestamp=ts)
    except Exception:  # pragma: no cover - defensive
        return None

    ara_dir.mkdir(parents=True, exist_ok=True)
    os.environ[LLM_TRACE_ROOT_ENV] = str(ara_dir)
    if stage is not None:
        os.environ[LLM_TRACE_STAGE_ENV] = stage
    return ara_dir


def deactivate_llm_tracing() -> None:
    """Unset tracer env vars. Idempotent; safe to call from ``finally`` blocks."""
    os.environ.pop(LLM_TRACE_ROOT_ENV, None)
    os.environ.pop(LLM_TRACE_STAGE_ENV, None)


def _find_tex_files(exp_dir: Path) -> list[str]:
    candidates = [exp_dir / "latex" / "template.tex", exp_dir / "template.tex"]
    return [str(p) for p in candidates if p.exists()]


def finalize_ara_for_idea(
    *,
    project_dir: str | Path,
    exp_dir: str | Path,
    idea: dict[str, Any],
    timestamp: str | None,
    bfts_config_path: str | None,
    model_spec: dict[str, Any],
    writing_profile: str | None,
    run_reexec: bool | None = None,
) -> ARAFinalizeResult:
    """Full ARA finalisation for one idea.

    Runs export → claim scan → optional re-execution, catching each stage's
    exceptions independently so a claim-scan blow-up doesn't lose the export.
    """
    result = ARAFinalizeResult()

    try:
        export = export_ara(
            project_dir=project_dir,
            exp_dir=exp_dir,
            idea=idea,
            timestamp=timestamp,
            bfts_config_path=bfts_config_path,
            model_spec=model_spec,
            writing_profile=writing_profile,
            provenance=_read_provenance_env(),
        )
    except Exception as exc:  # noqa: BLE001 — surface at boundary, don't crash pipeline
        result.error = f"export_ara failed: {exc}"
        logger.exception("export_ara failed")
        return result

    result.export = export
    exp_dir_path = Path(exp_dir)
    tex_files = _find_tex_files(exp_dir_path)
    if tex_files:
        try:
            claim_summary = write_claims_into_ara(
                ara_dir=export.root, tex_files=tex_files
            )
            update_manifest_claim_count(
                export.manifest_path,
                int(claim_summary.get("claim_count") or 0),
            )
            result.claim_summary = claim_summary
        except Exception as exc:  # noqa: BLE001
            logger.exception("write_claims_into_ara failed")
            # Not fatal — export is still valid, just without claim links.
            result.claim_summary = {"error": str(exc)}

    # Claim coverage runs regardless: even a producer that skipped tex
    # scanning gets an "unknown" report, which is more useful than silence.
    try:
        coverage = evaluate_claim_coverage(export.root)
        write_coverage_into_ara(export.root, coverage)
        result.claim_coverage = coverage
    except Exception as exc:  # noqa: BLE001
        logger.exception("evaluate_claim_coverage failed")

    do_reexec = reexec_enabled() if run_reexec is None else bool(run_reexec)
    if do_reexec:
        try:
            result.reexec_summary = reexec_ara(export.root)
        except Exception as exc:  # noqa: BLE001
            logger.exception("reexec_ara failed")
            result.reexec_summary = {"error": str(exc)}

    return result


def summarise_seed_stage(stage: SeedStageResult) -> str:
    """Human-readable one-liner for the CLI banner."""
    if stage.error:
        return f"❌ --seed-from-ara 无法加载: {stage.error}"
    if not stage.seed_used:
        return ""
    parent_id = stage.provenance.get("parent_node_id")
    parent_hash = stage.provenance.get("parent_content_hash")
    return (
        f"🌱 ARA seed staged: parent_node_id={parent_id} "
        f"parent_content_hash={parent_hash} → {stage.seed_path}"
    )


def summarise_finalize(idea_idx: int, result: ARAFinalizeResult) -> list[str]:
    """Return the log lines the pipeline should print for one idea's ARA."""
    lines: list[str] = []
    if result.error:
        lines.append(f"[想法 #{idea_idx}] ⚠️  ARA export 失败: {result.error}")
        return lines
    export = result.export
    if export is None:
        return lines
    if result.claim_summary and "error" not in result.claim_summary:
        lines.append(
            f"[想法 #{idea_idx}] ARA claims: "
            f"{result.claim_summary.get('claim_count')} scanned, "
            f"{result.claim_summary.get('resolved_count')} resolved"
        )
    elif result.claim_summary:  # error branch
        lines.append(
            f"[想法 #{idea_idx}] ⚠️  claim scan failed: {result.claim_summary.get('error')}"
        )
    if result.claim_coverage is not None:
        cov = result.claim_coverage
        lines.append(
            f"[想法 #{idea_idx}] ARA claim coverage: "
            f"score={cov.coverage_score:.2f} severity={cov.severity} "
            f"({cov.resolved_count}/{cov.claim_count})"
        )
    lines.append(
        f"[想法 #{idea_idx}] ARA export: {export.root} "
        f"(nodes={export.node_count}, missing={len(export.missing)})"
    )
    if result.reexec_summary:
        if "error" in result.reexec_summary:
            lines.append(
                f"[想法 #{idea_idx}] ⚠️  ARA re-exec 失败: {result.reexec_summary['error']}"
            )
        else:
            lines.append(
                f"[想法 #{idea_idx}] ARA re-exec: {result.reexec_summary.get('status')} "
                f"nodes={result.reexec_summary.get('verdict_count')} "
                f"report={result.reexec_summary.get('report_path')}"
            )
    return lines
