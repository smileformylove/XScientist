from __future__ import annotations

"""Pareto-front candidate pool for manuscript versions across reviewer dimensions.

Inspired by GEPA (https://github.com/gepa-ai/gepa) — instead of overwriting the
manuscript every round, we keep variants that excel on different reviewer
dimensions and let the next rewrite seed from a non-dominated ancestor.
"""

import hashlib
import os
import re
import shutil
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_scientist.utils.pipeline_contracts import (
    load_contract_artifact,
    save_contract_artifact,
)


_ARTIFACT_NAME = "manuscript_candidate_pool"
_POOL_DIR_NAME = "pareto_pool"
_CANDIDATES_SUBDIR = "candidates"
_ARCHIVED_SUBDIR = "archived"
_DEFAULT_MAX_SIZE = 20
_MIN_DIMS_FOR_ADMISSION = 3

# Order matches `ai_scientist.utils.self_review_optimizer._SCORE_KEYS`.
SCORE_KEYS: tuple[str, ...] = (
    "Originality",
    "Quality",
    "Clarity",
    "Significance",
    "Soundness",
    "Presentation",
    "Contribution",
)


def _now_iso() -> str:
    return datetime.now().isoformat()


def _pool_root(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / _POOL_DIR_NAME


def _candidates_dir(project_root: str | Path) -> Path:
    return _pool_root(project_root) / _CANDIDATES_SUBDIR


def _archived_dir(project_root: str | Path) -> Path:
    return _pool_root(project_root) / _ARCHIVED_SUBDIR


def _max_size_from_env(default: int = _DEFAULT_MAX_SIZE) -> int:
    raw = os.environ.get("AI_SCIENTIST_PARETO_POOL_MAX")
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(value, 1)


_COMMENT_RE = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def _normalize_latex(latex_text: str) -> str:
    """Strip line comments and blank lines so cosmetic edits don't shift the hash."""
    text = _COMMENT_RE.sub("", latex_text or "")
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def compute_candidate_key(latex_text: str) -> str:
    normalized = _normalize_latex(latex_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _impute_scores(raw_scores: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    present: dict[str, float] = {}
    for key in SCORE_KEYS:
        v = _coerce_score(raw_scores.get(key))
        if v is not None:
            present[key] = v
    if len(present) < _MIN_DIMS_FOR_ADMISSION:
        return present, []
    fill_value = statistics.median(present.values())
    imputed_dims: list[str] = []
    full: dict[str, float] = {}
    for key in SCORE_KEYS:
        if key in present:
            full[key] = present[key]
        else:
            full[key] = fill_value
            imputed_dims.append(key)
    return full, imputed_dims


def load_pareto_pool(project_root: str | Path) -> dict[str, Any]:
    payload = load_contract_artifact(project_root, _ARTIFACT_NAME, default=None)
    if not isinstance(payload, dict):
        return _empty_pool()
    payload.setdefault("schema_version", 1)
    payload.setdefault("candidates", [])
    payload.setdefault("front_keys", [])
    payload.setdefault("dominated_count", 0)
    payload.setdefault("evicted_count", 0)
    return payload


def _empty_pool() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": _now_iso(),
        "candidates": [],
        "front_keys": [],
        "dominated_count": 0,
        "evicted_count": 0,
    }


def save_pareto_pool(
    project_root: str | Path,
    payload: dict[str, Any],
    *,
    producer: str = "pareto_pool",
) -> str:
    payload["generated_at"] = _now_iso()
    return save_contract_artifact(
        project_root,
        _ARTIFACT_NAME,
        payload,
        producer=producer,
        depends_on=["review_state", "manuscript_state"],
    )


def _dominates(a_scores: dict[str, float], b_scores: dict[str, float]) -> bool:
    strictly_better = False
    for key in SCORE_KEYS:
        a = a_scores.get(key)
        b = b_scores.get(key)
        if a is None or b is None:
            return False
        if a < b:
            return False
        if a > b:
            strictly_better = True
    return strictly_better


def compute_pareto_front(candidates: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    valid = [c for c in candidates if isinstance(c, dict) and isinstance(c.get("scores"), dict)]
    for candidate in valid:
        scores = candidate["scores"]
        dominated = False
        for other in valid:
            if other is candidate:
                continue
            if _dominates(other.get("scores") or {}, scores):
                dominated = True
                break
        if not dominated:
            keys.append(str(candidate.get("key") or ""))
    return [k for k in keys if k]


def add_candidate(
    project_root: str | Path,
    *,
    latex_path: str | Path | None,
    scores: dict[str, Any],
    round_index: int,
    source: str = "review_round",
) -> dict[str, Any]:
    if latex_path is None:
        return {"status": "skipped_no_latex"}
    latex_path_obj = Path(latex_path).expanduser()
    if not latex_path_obj.exists():
        return {"status": "skipped_no_latex", "latex_path": str(latex_path_obj)}
    try:
        latex_text = latex_path_obj.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return {"status": "skipped_read_error", "reason": str(exc)}
    full_scores, imputed_dims = _impute_scores(scores or {})
    if not full_scores:
        return {"status": "skipped_low_signal", "present_dims": 0}
    if len(full_scores) - len(imputed_dims) < _MIN_DIMS_FOR_ADMISSION:
        return {
            "status": "skipped_low_signal",
            "present_dims": len(full_scores) - len(imputed_dims),
        }

    key = compute_candidate_key(latex_text)
    pool = load_pareto_pool(project_root)
    candidates: list[dict[str, Any]] = list(pool.get("candidates") or [])

    cand_dir = _candidates_dir(project_root)
    cand_dir.mkdir(parents=True, exist_ok=True)
    target_path = cand_dir / f"{key}.tex"
    if not target_path.exists():
        try:
            shutil.copyfile(latex_path_obj, target_path)
        except OSError as exc:
            return {"status": "skipped_copy_error", "reason": str(exc)}

    relpath = str(target_path.relative_to(Path(project_root).expanduser().resolve()))

    existing = next((c for c in candidates if c.get("key") == key), None)
    if existing is None:
        candidates.append({
            "key": key,
            "latex_relpath": relpath,
            "scores": full_scores,
            "imputed_dims": imputed_dims,
            "round_index": int(round_index),
            "source": str(source or "review_round"),
            "created_at": _now_iso(),
            "last_seen_round": int(round_index),
            "is_on_front": False,
            "times_seeded": 0,
        })
        admit_status = "added"
    else:
        existing.setdefault("round_index", int(round_index))
        existing["latex_relpath"] = relpath
        existing["last_seen_round"] = int(round_index)
        existing["scores"] = full_scores
        existing["imputed_dims"] = imputed_dims
        admit_status = "updated"

    front_keys = compute_pareto_front(candidates)
    front_set = set(front_keys)
    for candidate in candidates:
        candidate["is_on_front"] = candidate.get("key") in front_set

    pool["candidates"] = candidates
    pool["front_keys"] = front_keys
    pool["dominated_count"] = sum(1 for c in candidates if not c.get("is_on_front"))
    save_pareto_pool(project_root, pool)
    return {"status": admit_status, "key": key, "is_on_front": key in front_set}


def evict_dominated(
    project_root: str | Path,
    *,
    max_size: int | None = None,
) -> int:
    cap = max_size if max_size is not None else _max_size_from_env()
    pool = load_pareto_pool(project_root)
    candidates: list[dict[str, Any]] = list(pool.get("candidates") or [])
    if not candidates:
        return 0

    front_keys = set(compute_pareto_front(candidates))
    archived_dir = _archived_dir(project_root)
    archived_dir.mkdir(parents=True, exist_ok=True)
    project_root_path = Path(project_root).expanduser().resolve()
    evicted = 0

    def _archive(candidate: dict[str, Any]) -> None:
        nonlocal evicted
        relpath = candidate.get("latex_relpath")
        if not relpath:
            return
        src = project_root_path / relpath
        if src.exists():
            try:
                shutil.move(str(src), str(archived_dir / src.name))
            except OSError:
                return
        evicted += 1

    front_members = [c for c in candidates if c.get("key") in front_keys]
    dominated_members = [c for c in candidates if c.get("key") not in front_keys]
    dominated_members.sort(key=lambda c: (int(c.get("round_index") or 0), str(c.get("created_at") or "")))

    while len(front_members) + len(dominated_members) > cap and dominated_members:
        victim = dominated_members.pop(0)
        _archive(victim)

    if len(front_members) > cap:
        front_members.sort(
            key=lambda c: (
                statistics.mean(c.get("scores", {}).values()) if c.get("scores") else 0.0,
                -int(c.get("round_index") or 0),
            ),
            reverse=True,
        )
        kept_front = front_members[:cap]
        for victim in front_members[cap:]:
            _archive(victim)
        front_members = kept_front

    survivors = front_members + dominated_members
    new_front_keys = compute_pareto_front(survivors)
    new_front_set = set(new_front_keys)
    for candidate in survivors:
        candidate["is_on_front"] = candidate.get("key") in new_front_set

    pool["candidates"] = survivors
    pool["front_keys"] = new_front_keys
    pool["dominated_count"] = sum(1 for c in survivors if not c.get("is_on_front"))
    pool["evicted_count"] = int(pool.get("evicted_count") or 0) + evicted
    save_pareto_pool(project_root, pool)
    return evicted


def select_seed_for_next_round(
    project_root: str | Path,
    *,
    strategy: str = "weighted_pareto",
    avoid_keys: list[str] | None = None,
) -> dict[str, Any] | None:
    pool = load_pareto_pool(project_root)
    candidates: list[dict[str, Any]] = pool.get("candidates") or []
    front_keys = set(pool.get("front_keys") or [])
    front = [c for c in candidates if c.get("key") in front_keys]
    if not front:
        return None
    avoid = {str(k) for k in (avoid_keys or [])}
    pool_root = Path(project_root).expanduser().resolve()

    def _weight(candidate: dict[str, Any]) -> tuple[float, int]:
        scores = candidate.get("scores") or {}
        weak_dims = sum(1 for v in scores.values() if float(v or 0) < 7.0)
        seeded = int(candidate.get("times_seeded") or 0)
        return float(weak_dims), -seeded

    pool_keys_seen = [c for c in front if str(c.get("key")) not in avoid]
    if not pool_keys_seen:
        pool_keys_seen = front
    pool_keys_seen.sort(key=_weight, reverse=True)
    chosen = pool_keys_seen[0]
    chosen["times_seeded"] = int(chosen.get("times_seeded") or 0) + 1

    save_pareto_pool(project_root, pool)
    latex_relpath = chosen.get("latex_relpath")
    latex_abspath = str(pool_root / latex_relpath) if latex_relpath else None
    return {
        "key": chosen.get("key"),
        "latex_path": latex_abspath,
        "scores": chosen.get("scores"),
        "imputed_dims": chosen.get("imputed_dims"),
        "round_index": chosen.get("round_index"),
        "source": chosen.get("source"),
        "strategy": strategy,
    }


def pareto_pool_enabled() -> bool:
    return str(os.environ.get("AI_SCIENTIST_PARETO_POOL") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def maybe_select_seed_path(
    project_root: str | Path,
    *,
    avoid_keys: list[str] | None = None,
) -> str | None:
    """Return a seed LaTeX absolute path when the pool is enabled and has a usable front member.

    Returns None on any failure (pool off, empty front, missing file, exception) so callers
    can fall back to the default LaTeX without special-casing.
    """
    if not pareto_pool_enabled():
        return None
    try:
        seed = select_seed_for_next_round(project_root, avoid_keys=avoid_keys)
    except Exception:
        return None
    if not isinstance(seed, dict):
        return None
    path = seed.get("latex_path")
    if not path:
        return None
    try:
        return path if Path(path).exists() else None
    except OSError:
        return None


def merge_enabled() -> bool:
    return str(os.environ.get("AI_SCIENTIST_PARETO_MERGE") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def select_merge_pair(
    project_root: str | Path,
    *,
    min_complementarity: float = 1.5,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pick two front members whose strengths are most complementary.

    Complementarity = sum over dims of max(0, A[dim] - B[dim]) + max(0, B[dim] - A[dim]).
    A returns None if no front pair meets the threshold (defaults to 1.5 — roughly 0.2/dim).
    """
    pool = load_pareto_pool(project_root)
    candidates: list[dict[str, Any]] = pool.get("candidates") or []
    front_keys = set(pool.get("front_keys") or [])
    front = [c for c in candidates if c.get("key") in front_keys and isinstance(c.get("scores"), dict)]
    if len(front) < 2:
        return None
    best: tuple[float, dict[str, Any], dict[str, Any]] | None = None
    for i, a in enumerate(front):
        for b in front[i + 1 :]:
            score = 0.0
            for dim in SCORE_KEYS:
                av = float((a.get("scores") or {}).get(dim) or 0.0)
                bv = float((b.get("scores") or {}).get(dim) or 0.0)
                score += abs(av - bv)
            if best is None or score > best[0]:
                best = (score, a, b)
    if best is None or best[0] < min_complementarity:
        return None
    return best[1], best[2]


def describe_merge_pair(pair: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    """Distill the complementary-strengths picture into a structured payload for LLM prompts.

    Returns:
        {
          "candidate_a": {key, latex_relpath, scores, strengths: [dim,...]},
          "candidate_b": {...},
          "complementary_dims": [{dim, a, b, winner}, ...],
        }
    """
    a, b = pair
    a_scores = a.get("scores") or {}
    b_scores = b.get("scores") or {}
    complementary: list[dict[str, Any]] = []
    for dim in SCORE_KEYS:
        av = float(a_scores.get(dim) or 0.0)
        bv = float(b_scores.get(dim) or 0.0)
        if av == bv:
            continue
        complementary.append(
            {
                "dim": dim,
                "a": av,
                "b": bv,
                "winner": "a" if av > bv else "b",
                "gap": round(abs(av - bv), 3),
            }
        )
    complementary.sort(key=lambda row: row["gap"], reverse=True)

    def _strengths(scores: dict[str, Any], threshold: float = 7.5) -> list[str]:
        return [dim for dim in SCORE_KEYS if float(scores.get(dim) or 0.0) >= threshold]

    return {
        "candidate_a": {
            "key": a.get("key"),
            "latex_relpath": a.get("latex_relpath"),
            "scores": {dim: float(a_scores.get(dim) or 0.0) for dim in SCORE_KEYS},
            "strengths": _strengths(a_scores),
        },
        "candidate_b": {
            "key": b.get("key"),
            "latex_relpath": b.get("latex_relpath"),
            "scores": {dim: float(b_scores.get(dim) or 0.0) for dim in SCORE_KEYS},
            "strengths": _strengths(b_scores),
        },
        "complementary_dims": complementary,
    }


def maybe_build_merge_advisory(project_root: str | Path) -> dict[str, Any] | None:
    """Env-gated helper for callers — returns the merge-pair description or None."""
    if not merge_enabled():
        return None
    pair = select_merge_pair(project_root)
    if pair is None:
        return None
    return describe_merge_pair(pair)
