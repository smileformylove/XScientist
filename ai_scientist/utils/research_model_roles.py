"""Fail-closed scientific-authority boundaries for project model routes."""

from __future__ import annotations

from argparse import Namespace

from ai_scientist.utils.provider_registry import resolve_model_provider

_JUDGMENT_ROUTES = (
    ("ideation", "--model-ideation", "model_ideation", None),
    ("final figure selection", "--model-agg-plots", "model_agg_plots", None),
    ("source attribution", "--model-citation", "model_citation", None),
    ("paper review", "--model-review", "model_review", None),
    ("candidate ranking", "--idea-rank-model", "idea_rank_model", "model_writeup"),
    ("quality judgment", "--quality-model", "quality_model", "model_review"),
)


def _contains_glm53(value: object) -> bool:
    for selected in (item.strip() for item in str(value or "").split(",")):
        if not selected:
            continue
        try:
            spec = resolve_model_provider(selected)
        except ValueError:
            # Normal provider/model validation remains responsible for malformed
            # routes; this helper only enforces the GLM scientific-role boundary.
            continue
        if str(spec.client_model).casefold() == "glm-5.3":
            return True
    return False


def glm53_judgment_role_violations(args: Namespace) -> list[tuple[str, str]]:
    """Return project judgment roles that currently resolve to GLM-5.3."""

    violations: list[tuple[str, str]] = []
    for role, flag, attribute, fallback_attribute in _JUDGMENT_ROUTES:
        value = getattr(args, attribute, None)
        if not str(value or "").strip() and fallback_attribute:
            value = getattr(args, fallback_attribute, None)
        if _contains_glm53(value):
            violations.append((role, flag))
    return violations


def enforce_glm53_project_role_boundary(args: Namespace) -> None:
    """Reject GLM-5.3 in project-level scientific judgment routes.

    GLM may implement an already judgment-locked BFTS task, generate code, use
    bounded tools, report raw execution outcomes, and draft non-authoritative prose.
    It may not select the
    research question, final figures, source attribution, ranked candidate, quality
    result, or review.
    """

    violations = glm53_judgment_role_violations(args)
    if not violations:
        return
    rendered = ", ".join(f"{role} ({flag})" for role, flag in violations)
    recovery_flags = " ".join(dict.fromkeys(flag for _role, flag in violations))
    raise ValueError(
        "GLM-5.3 cannot own project scientific-judgment routes: "
        f"{rendered}. Keep GLM for locked-task implementation, bounded execution, "
        f"and non-authoritative drafting; set non-GLM model values with "
        f"{recovery_flags}."
    )


__all__ = [
    "enforce_glm53_project_role_boundary",
    "glm53_judgment_role_violations",
]
