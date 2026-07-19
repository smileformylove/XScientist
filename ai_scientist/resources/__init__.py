from __future__ import annotations

from importlib.resources import files
from pathlib import Path

_CONFIG_NAMES = {
    "default": "bfts_default.yaml",
    "deep": "bfts_deep.yaml",
}

_LATEX_TEMPLATE_NAMES = {
    "icbinb": "blank_icbinb_latex",
    "icml": "blank_icml_latex",
    "normal": "blank_icml_latex",
    "journal": "blank_icml_latex",
    "extended": "blank_icbinb_latex",
}


def bfts_config_path(profile: str = "default") -> Path:
    """Return the installed BFTS configuration for a supported profile."""

    normalized = str(profile or "default").strip().lower()
    try:
        filename = _CONFIG_NAMES[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(_CONFIG_NAMES))
        raise ValueError(
            f"Unknown BFTS config profile {profile!r}; expected one of: {choices}"
        ) from exc
    resource = files("ai_scientist.resources.configs").joinpath(filename)
    path = Path(str(resource))
    if not path.is_file():
        raise FileNotFoundError(f"Packaged BFTS config is missing: {path}")
    return path


def resolve_bfts_config_path(
    value: str | Path | None = None, *, base_dir: str | Path | None = None
) -> Path:
    """Resolve an explicit path or a packaged ``default``/``deep`` profile."""

    if value is None:
        return bfts_config_path("default")
    text = str(value).strip()
    profile_aliases = {
        "default": "default",
        "bfts_config.yaml": "default",
        "deep": "deep",
        "bfts_config_deep.yaml": "deep",
    }
    candidate = Path(text).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_absolute():
        relative_candidate = Path(base_dir or Path.cwd()).expanduser() / candidate
        if relative_candidate.is_file():
            return relative_candidate.resolve()
    profile = profile_aliases.get(text.lower()) or profile_aliases.get(
        candidate.name.lower()
    )
    if profile is not None:
        return bfts_config_path(profile)
    raise FileNotFoundError(
        f"BFTS config not found: {value}. Use an existing YAML path or "
        "the profile name 'default'/'deep'."
    )


def package_root() -> Path:
    """Return the installed ``ai_scientist`` package directory."""

    return Path(__file__).resolve().parent.parent


def latex_template_dir(template: str) -> Path:
    """Return a packaged LaTeX template directory."""

    normalized = str(template or "").strip().lower()
    directory_name = _LATEX_TEMPLATE_NAMES.get(normalized, template)
    path = package_root() / str(directory_name)
    if not path.is_dir():
        choices = ", ".join(sorted(_LATEX_TEMPLATE_NAMES))
        raise ValueError(
            f"Unknown LaTeX template {template!r}; expected one of: {choices}"
        )
    return path


def idea_resource_path(filename: str = "i_cant_believe_its_not_better.json") -> Path:
    """Return a packaged example idea or workshop-description file."""

    name = Path(filename).name
    path = package_root() / "ideas" / name
    if not path.is_file():
        raise FileNotFoundError(f"Packaged idea resource is missing: {name}")
    return path


__all__ = [
    "bfts_config_path",
    "idea_resource_path",
    "latex_template_dir",
    "package_root",
    "resolve_bfts_config_path",
]
