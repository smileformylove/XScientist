from __future__ import annotations

from argparse import Namespace

import pytest

from ai_scientist.utils.research_model_roles import (
    enforce_glm53_project_role_boundary,
    glm53_judgment_role_violations,
)


def _args(**overrides: str | None) -> Namespace:
    values: dict[str, str | None] = {
        "model_ideation": "openai/gpt-4.1",
        "model_agg_plots": "openai/gpt-4.1",
        "model_citation": "openai/gpt-4.1",
        "model_writeup": "openai_compat/glm-5.3",
        "model_review": "openai/gpt-4.1",
        "idea_rank_model": "openai/gpt-4.1",
        "quality_model": "openai/gpt-4.1",
    }
    values.update(overrides)
    return Namespace(**values)


def test_glm53_is_allowed_for_drafting_when_judgment_routes_are_separate() -> None:
    args = _args()
    assert glm53_judgment_role_violations(args) == []
    enforce_glm53_project_role_boundary(args)


@pytest.mark.parametrize(
    ("attribute", "role"),
    [
        ("model_ideation", "ideation"),
        ("model_agg_plots", "final figure selection"),
        ("model_citation", "source attribution"),
        ("model_review", "paper review"),
        ("idea_rank_model", "candidate ranking"),
        ("quality_model", "quality judgment"),
    ],
)
def test_glm53_is_rejected_from_every_project_judgment_route(
    attribute: str,
    role: str,
) -> None:
    args = _args(**{attribute: "custom/glm-5.3"})
    with pytest.raises(ValueError, match=role):
        enforce_glm53_project_role_boundary(args)


def test_implicit_ranking_and_quality_fallbacks_are_checked() -> None:
    args = _args(
        idea_rank_model=None,
        model_writeup="openai_compat/glm-5.3",
        quality_model=None,
        model_review="openai_compat/glm-5.3",
    )
    violations = glm53_judgment_role_violations(args)
    assert ("candidate ranking", "--idea-rank-model") in violations
    assert ("quality judgment", "--quality-model") in violations
