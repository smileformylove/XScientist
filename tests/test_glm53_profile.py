from __future__ import annotations

from collections.abc import Mapping

from omegaconf import OmegaConf

from ai_scientist.apps.validate import _validate_glm53_profile
from ai_scientist.resources import bfts_config_path, resolve_bfts_config_path

GLM53_MODEL = "openai_compat/glm-5.3"
JUDGMENT_REQUIRED = "__xscientist_non_glm_judgment_model_required__"


def _load_profile() -> dict:
    payload = OmegaConf.to_container(
        OmegaConf.load(bfts_config_path("glm53")), resolve=True
    )
    assert isinstance(payload, dict)
    return payload


def _all_keys(value: object) -> list[str]:
    if isinstance(value, Mapping):
        return [str(key).lower() for key in value] + [
            nested for child in value.values() for nested in _all_keys(child)
        ]
    if isinstance(value, list):
        return [nested for child in value for nested in _all_keys(child)]
    return []


def test_glm53_profile_aliases_resolve_to_packaged_resource() -> None:
    expected = bfts_config_path("glm53")

    assert expected.is_file()
    assert resolve_bfts_config_path("glm53") == expected
    assert resolve_bfts_config_path("glm-5.3") == expected
    assert resolve_bfts_config_path("bfts_glm53.yaml") == expected
    assert resolve_bfts_config_path("bfts_config_glm53.yaml") == expected


def test_glm53_profile_reserves_scientific_roles_for_explicit_judgment() -> None:
    profile = _load_profile()
    default = OmegaConf.to_container(
        OmegaConf.load(bfts_config_path("default")), resolve=True
    )
    assert isinstance(default, dict)
    agent = profile["agent"]
    assert agent["code"]["model"] == GLM53_MODEL
    assert profile["report"]["model"] == GLM53_MODEL
    for role in (
        "judgment",
        "feedback",
        "vlm_feedback",
        "summary",
        "select_node",
    ):
        assert agent[role]["model"] == JUDGMENT_REQUIRED
    assert profile["report"]["temp"] == default["report"]["temp"]
    for role in ("code", "feedback", "vlm_feedback"):
        assert agent[role]["temp"] == default["agent"][role]["temp"]
        assert agent[role]["max_tokens"] == default["agent"][role]["max_tokens"]
    assert agent["summary"]["temp"] == default["report"]["temp"]
    assert agent["summary"]["max_tokens"] is None
    assert agent["judgment"]["max_tokens"] == 4096
    assert agent["select_node"]["max_tokens"] == 4096


def test_glm53_profile_contains_no_connection_or_secret_material() -> None:
    profile = _load_profile()
    forbidden_fragments = (
        "api_key",
        "apikey",
        "authorization",
        "base_url",
        "endpoint",
        "headers",
    )

    assert not {
        key
        for key in _all_keys(profile)
        if any(fragment in key for fragment in forbidden_fragments)
    }
    serialized = OmegaConf.to_yaml(OmegaConf.create(profile)).lower()
    assert "sk-" not in serialized
    assert "http://" not in serialized
    assert "https://" not in serialized


def test_glm53_profile_is_isolated_held_out_and_bounded() -> None:
    profile = _load_profile()
    execution = profile["exec"]
    budget = profile["llm_budget"]
    agent = profile["agent"]
    seeds = agent["multi_seed_eval"]["seeds"]

    assert execution["require_isolation"] is True
    assert execution["network"] == "none"
    assert execution["allow_experiment_network"] is False
    assert execution["read_only_root"] is True
    assert execution["timeout"] == 3600

    assert budget["max_total_tokens"] == 500_000
    assert budget["max_wall_time_seconds"] == 21_600
    assert agent["multi_seed_eval"]["num_seeds"] == len(seeds)
    assert len(seeds) >= 3
    assert len(seeds) == len(set(seeds))
    assert 42 not in seeds

    assert agent["code"]["max_tokens"] == 8192
    assert agent["feedback"]["max_tokens"] == 4096
    assert 0 < agent["num_workers"] <= 64
    assert all(0 < value <= 32 for value in agent["stages"].values())
    _validate_glm53_profile(profile)


def test_installed_profile_validator_rejects_authority_or_transport_drift() -> None:
    profile = _load_profile()
    profile["agent"]["select_node"] = {"model": GLM53_MODEL}

    try:
        _validate_glm53_profile(profile)
    except RuntimeError as exc:
        assert "model routing" in str(exc)
    else:
        raise AssertionError("select_node authority drift was accepted")

    profile = _load_profile()
    profile["agent"]["planner"] = {"model": GLM53_MODEL}
    try:
        _validate_glm53_profile(profile)
    except RuntimeError as exc:
        assert "model routing" in str(exc)
    else:
        raise AssertionError("unapproved planner-model authority drift was accepted")

    profile = _load_profile()
    profile["agent"]["code"]["endpoint"] = "https://invalid.example/v1"
    try:
        _validate_glm53_profile(profile)
    except RuntimeError as exc:
        assert "transport data" in str(exc)
    else:
        raise AssertionError("embedded transport data was accepted")

    profile = _load_profile()
    profile["llm_budget"]["max_total_tokens"] = 1
    try:
        _validate_glm53_profile(profile)
    except RuntimeError as exc:
        assert "budget contract" in str(exc)
    else:
        raise AssertionError("budget contract drift was accepted")

    profile = _load_profile()
    profile["agent"]["evidence_gate"]["stage3_min_improved"] = 1
    try:
        _validate_glm53_profile(profile)
    except RuntimeError as exc:
        assert "evidence gate" in str(exc)
    else:
        raise AssertionError("deterministic evidence gate drift was accepted")

    profile = _load_profile()
    profile["agent"]["multi_seed_eval"]["max_relative_ci_half_width"] = 10.0
    try:
        _validate_glm53_profile(profile)
    except RuntimeError as exc:
        assert "uncertainty gate" in str(exc)
    else:
        raise AssertionError("confirmation uncertainty gate drift was accepted")
