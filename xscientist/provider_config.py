from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text
from ai_scientist.utils.provider_registry import resolve_model_provider

CONFIG_SCHEMA_VERSION = 1
CONFIG_RELATIVE_PATH = Path(".xscientist") / "providers.json"
DEFAULT_ENV_FILE = ".env"
DEFAULT_MODELS = {"zhipu": "glm-4-flash"}
PLACEHOLDER_VALUES = {"", "replace-me", "your-api-key", "your_api_key_here"}


@dataclass(frozen=True)
class ProviderField:
    name: str
    secret: bool = True
    required: bool = True
    default: str | None = None
    aliases: tuple[str, ...] = ()


PROVIDER_FIELDS: dict[str, tuple[ProviderField, ...]] = {
    "zhipu": (ProviderField("ZHIPU_API_KEY"),),
    "openai": (
        ProviderField("OPENAI_API_KEY"),
        ProviderField("OPENAI_ORG_ID", secret=False, required=False),
        ProviderField("OPENAI_PROJECT", secret=False, required=False),
    ),
    "anthropic": (ProviderField("ANTHROPIC_API_KEY"),),
    "deepseek": (ProviderField("DEEPSEEK_API_KEY"),),
    "gemini": (ProviderField("GEMINI_API_KEY", aliases=("GOOGLE_API_KEY",)),),
    "openrouter": (ProviderField("OPENROUTER_API_KEY"),),
    "huggingface": (ProviderField("HUGGINGFACE_API_KEY"),),
    "ollama": (
        ProviderField(
            "OLLAMA_BASE_URL",
            secret=False,
            required=False,
            default="http://localhost:11434/v1",
        ),
        ProviderField("OLLAMA_API_KEY", required=False),
    ),
    "openai_compat": (
        ProviderField("OPENAI_COMPAT_API_KEY", aliases=("OPENAI_API_KEY",)),
        ProviderField(
            "OPENAI_COMPAT_BASE_URL", secret=False, aliases=("OPENAI_BASE_URL",)
        ),
    ),
    "bedrock": (
        ProviderField("AWS_ACCESS_KEY_ID"),
        ProviderField("AWS_SECRET_ACCESS_KEY"),
        ProviderField("AWS_REGION_NAME", secret=False),
        ProviderField("AWS_SESSION_TOKEN", required=False),
    ),
    "vertex_ai": (
        ProviderField("GOOGLE_APPLICATION_CREDENTIALS", secret=False, required=False),
        ProviderField("GOOGLE_CLOUD_PROJECT", secret=False, required=False),
        ProviderField("CLOUD_ML_REGION", secret=False, required=False),
    ),
}
PROVIDER_NAMES = tuple(PROVIDER_FIELDS)
ALLOWED_ENV_NAMES = {
    name
    for fields in PROVIDER_FIELDS.values()
    for field in fields
    for name in (field.name, *field.aliases)
} | {
    "RESEARCH_OUTPUT_DIR",
    "S2_API_KEY",
    "AI_SCIENTIST_MODEL_IDEATION",
    "AI_SCIENTIST_MODEL_AGG_PLOTS",
    "AI_SCIENTIST_MODEL_WRITEUP",
    "AI_SCIENTIST_MODEL_WRITEUP_SMALL",
    "AI_SCIENTIST_MODEL_CITATION",
    "AI_SCIENTIST_MODEL_REVIEW",
}


class ProviderConfigError(ValueError):
    """Raised when provider configuration is unsafe or malformed."""


def workspace_config_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / CONFIG_RELATIVE_PATH


def resolve_env_file(root: str | Path, env_file: str) -> Path:
    workspace = Path(root).expanduser().resolve()
    relative = Path(str(env_file or DEFAULT_ENV_FILE).strip())
    if relative.is_absolute():
        raise ProviderConfigError("provider env_file must be relative to the workspace")
    target = (workspace / relative).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ProviderConfigError(
            "provider env_file cannot escape the workspace"
        ) from exc
    if target == workspace_config_path(workspace):
        raise ProviderConfigError("provider env_file cannot replace provider metadata")
    return target


def discover_workspace_root() -> Path | None:
    explicit = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    candidate = Path(explicit).expanduser().resolve() if explicit else Path.cwd()
    return candidate if workspace_config_path(candidate).is_file() else None


def empty_provider_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "active_provider": None,
        "env_file": DEFAULT_ENV_FILE,
        "providers": {},
    }


def load_provider_config(
    root: str | Path,
    *,
    missing_ok: bool = True,
) -> dict[str, Any]:
    path = workspace_config_path(root)
    if not path.is_file():
        if missing_ok:
            return empty_provider_config()
        raise ProviderConfigError(f"provider configuration not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProviderConfigError(f"cannot read provider configuration: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderConfigError("provider configuration must be a JSON object")
    unknown_top_level = set(payload) - {
        "schema_version",
        "active_provider",
        "env_file",
        "providers",
    }
    if unknown_top_level:
        raise ProviderConfigError(
            f"provider configuration has unknown fields: {sorted(unknown_top_level)}"
        )
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ProviderConfigError("unsupported provider configuration schema")
    if not isinstance(payload.get("providers"), dict):
        raise ProviderConfigError("provider configuration has invalid providers")
    env_file = payload.get("env_file")
    if not isinstance(env_file, str) or not env_file.strip():
        raise ProviderConfigError("provider configuration has invalid env_file")
    resolve_env_file(root, env_file)
    providers = payload["providers"]
    for provider, entry in providers.items():
        if provider not in PROVIDER_FIELDS or not isinstance(entry, dict):
            raise ProviderConfigError(
                f"provider configuration has invalid provider entry: {provider!r}"
            )
        unknown_entry_fields = set(entry) - {"model", "credential_env_vars"}
        if unknown_entry_fields:
            raise ProviderConfigError(
                f"provider configuration has unknown fields for {provider!r}: "
                f"{sorted(unknown_entry_fields)}"
            )
        try:
            validate_provider_model(provider, entry.get("model"))
        except ProviderConfigError as exc:
            raise ProviderConfigError(
                f"provider configuration has invalid model for {provider!r}: {exc}"
            ) from exc
        credential_names = entry.get("credential_env_vars")
        if not isinstance(credential_names, list) or any(
            not isinstance(name, str) or name not in ALLOWED_ENV_NAMES
            for name in credential_names
        ):
            raise ProviderConfigError(
                f"provider configuration has invalid credential fields for {provider!r}"
            )
    active = payload.get("active_provider")
    if active is not None and active not in providers:
        raise ProviderConfigError(
            "provider configuration active_provider is not configured"
        )
    return payload


def provider_config_payload(
    *,
    provider: str,
    model: str,
    env_file: str = DEFAULT_ENV_FILE,
) -> dict[str, Any]:
    normalized_provider, selected_model = validate_provider_model(provider, model)
    normalized_env_file = Path(env_file or DEFAULT_ENV_FILE).as_posix()
    if (
        Path(normalized_env_file).is_absolute()
        or ".." in Path(normalized_env_file).parts
    ):
        raise ProviderConfigError("provider env_file must stay inside the workspace")
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "active_provider": normalized_provider,
        "env_file": normalized_env_file,
        "providers": {
            normalized_provider: {
                "model": selected_model,
                "credential_env_vars": [
                    name
                    for field in PROVIDER_FIELDS[normalized_provider]
                    for name in (field.name, *field.aliases)
                ],
            }
        },
    }


def validate_provider_model(provider: str, model: str | None) -> tuple[str, str]:
    normalized = str(provider or "").strip().lower()
    if normalized not in PROVIDER_FIELDS:
        choices = ", ".join(PROVIDER_NAMES)
        raise ProviderConfigError(
            f"unknown provider {provider!r}; expected one of: {choices}"
        )
    selected = str(model or DEFAULT_MODELS.get(normalized) or "").strip()
    if not selected:
        raise ProviderConfigError(
            f"--model is required for provider {normalized!r}; use a provider-prefixed "
            f"ID such as {normalized}/<model>"
        )
    spec = resolve_model_provider(selected)
    if spec.provider != normalized:
        raise ProviderConfigError(
            f"model {selected!r} resolves to provider {spec.provider!r}, not "
            f"{normalized!r}"
        )
    return normalized, selected


def _clean_env_value(raw: str) -> str:
    value = str(raw).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value


def read_env_file(path: str | Path) -> dict[str, str]:
    target = Path(path)
    if not target.is_file():
        return {}
    values: dict[str, str] = {}
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, raw = stripped.split("=", 1)
        key = name.strip()
        if key in ALLOWED_ENV_NAMES:
            values[key] = _clean_env_value(raw)
    return values


def _is_configured_value(value: str | None) -> bool:
    return str(value or "").strip().lower() not in PLACEHOLDER_VALUES


def configured_field_value(
    field: ProviderField,
    stored: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
) -> str:
    for source in (environ if environ is not None else os.environ, stored):
        for name in (field.name, *field.aliases):
            value = str(source.get(name) or "").strip()
            if _is_configured_value(value):
                return value
    return str(field.default or "").strip()


def _env_file_is_private(path: Path) -> bool:
    if os.name == "nt" or not path.is_file():
        return True
    return path.stat().st_mode & 0o077 == 0


def load_workspace_environment(root: str | Path | None = None) -> dict[str, Any]:
    workspace = (
        Path(root).expanduser().resolve()
        if root is not None
        else discover_workspace_root()
    )
    if workspace is None:
        return {"loaded": False, "workspace": None, "loaded_names": []}
    config = load_provider_config(workspace, missing_ok=False)
    env_file = resolve_env_file(
        workspace, str(config.get("env_file") or DEFAULT_ENV_FILE)
    )
    if env_file.is_file() and not _env_file_is_private(env_file):
        return {
            "loaded": False,
            "workspace": str(workspace),
            "loaded_names": [],
            "error": f"refusing to load credentials with broad permissions: {env_file}",
        }
    loaded_names: list[str] = []
    for name, value in read_env_file(env_file).items():
        if _is_configured_value(value) and name not in os.environ:
            os.environ[name] = value
            loaded_names.append(name)
    active = str(config.get("active_provider") or "").strip()
    entry = config.get("providers", {}).get(active, {}) if active else {}
    model = str(entry.get("model") or "").strip() if isinstance(entry, dict) else ""
    if active:
        os.environ.setdefault("AI_SCIENTIST_ACTIVE_PROVIDER", active)
    if model:
        os.environ.setdefault("AI_SCIENTIST_DEFAULT_MODEL", model)
        os.environ.setdefault("ZHIPU_DEFAULT_MODEL", model)
    return {
        "loaded": True,
        "workspace": str(workspace),
        "active_provider": active or None,
        "model": model or None,
        "loaded_names": sorted(loaded_names),
    }


def _secure_atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        try:
            os.fchmod(descriptor, 0o600)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)


def update_env_file(path: str | Path, updates: Mapping[str, str]) -> None:
    target = Path(path)
    invalid = set(updates) - ALLOWED_ENV_NAMES
    if invalid:
        raise ProviderConfigError(f"unsupported environment fields: {sorted(invalid)}")
    for name, value in updates.items():
        if "\n" in value or "\r" in value:
            raise ProviderConfigError(
                f"environment value for {name} contains a newline"
            )
    original_lines = (
        target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    )
    remaining = dict(updates)
    output: list[str] = []
    for line in original_lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            name = stripped.split("=", 1)[0].strip()
            if name in remaining:
                output.append(f"{name}={remaining.pop(name)}")
                continue
        output.append(line)
    if remaining and output and output[-1]:
        output.append("")
    output.extend(f"{name}={value}" for name, value in remaining.items())
    _secure_atomic_write(target, "\n".join(output).rstrip() + "\n")


def save_provider(
    root: str | Path,
    *,
    provider: str,
    model: str,
    env_file: str = DEFAULT_ENV_FILE,
    activate: bool = True,
) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    normalized, selected_model = validate_provider_model(provider, model)
    config = load_provider_config(workspace, missing_ok=False)
    env_path = resolve_env_file(workspace, env_file)
    providers = config.setdefault("providers", {})
    providers[normalized] = {
        "model": selected_model,
        "credential_env_vars": [
            name
            for field in PROVIDER_FIELDS[normalized]
            for name in (field.name, *field.aliases)
        ],
    }
    config["env_file"] = env_path.relative_to(workspace).as_posix()
    if activate:
        config["active_provider"] = normalized
    atomic_write_json(workspace_config_path(workspace), config, indent=2)
    return config


def activate_provider(root: str | Path, provider: str) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    normalized = str(provider or "").strip().lower()
    config = load_provider_config(workspace, missing_ok=False)
    if normalized not in config["providers"]:
        raise ProviderConfigError(f"provider {normalized!r} is not configured")
    config["active_provider"] = normalized
    atomic_write_json(workspace_config_path(workspace), config, indent=2)
    return config


def remove_provider(root: str | Path, provider: str) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    normalized = str(provider or "").strip().lower()
    config = load_provider_config(workspace, missing_ok=False)
    if normalized not in config["providers"]:
        raise ProviderConfigError(f"provider {normalized!r} is not configured")
    del config["providers"][normalized]
    if config.get("active_provider") == normalized:
        config["active_provider"] = next(iter(config["providers"]), None)
    atomic_write_json(workspace_config_path(workspace), config, indent=2)
    return config


def update_bfts_models(path: str | Path, model: str) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    try:
        payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderConfigError(f"cannot read BFTS config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderConfigError("BFTS configuration must be a mapping")
    payload.setdefault("report", {})["model"] = model
    agent = payload.setdefault("agent", {})
    for key in ("code", "feedback", "vlm_feedback"):
        agent.setdefault(key, {})["model"] = model
    atomic_write_text(
        target,
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    return True


def provider_statuses(root: str | Path) -> list[dict[str, Any]]:
    workspace = Path(root).expanduser().resolve()
    config = load_provider_config(workspace, missing_ok=False)
    env_file = resolve_env_file(
        workspace, str(config.get("env_file") or DEFAULT_ENV_FILE)
    )
    env_error = (
        f"credential file has broad permissions: {env_file}"
        if env_file.is_file() and not _env_file_is_private(env_file)
        else ""
    )
    stored = read_env_file(env_file)
    active = str(config.get("active_provider") or "")
    configured_entries = config.get("providers", {})
    rows = []
    for provider in PROVIDER_NAMES:
        entry = configured_entries.get(provider, {})
        fields = PROVIDER_FIELDS[provider]
        missing = [
            " | ".join((field.name, *field.aliases))
            for field in fields
            if field.required
            and not _is_configured_value(configured_field_value(field, stored))
        ]
        rows.append(
            {
                "provider": provider,
                "active": provider == active,
                "configured": isinstance(entry, dict) and bool(entry),
                "credentials_available": not missing,
                "ready": (
                    isinstance(entry, dict)
                    and bool(entry)
                    and not missing
                    and not env_error
                ),
                "model": (
                    str(entry.get("model") or "") if isinstance(entry, dict) else ""
                ),
                "missing": missing,
                "error": env_error,
            }
        )
    return rows


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "DEFAULT_MODELS",
    "PROVIDER_FIELDS",
    "PROVIDER_NAMES",
    "ProviderConfigError",
    "ProviderField",
    "activate_provider",
    "configured_field_value",
    "discover_workspace_root",
    "load_provider_config",
    "load_workspace_environment",
    "provider_config_payload",
    "provider_statuses",
    "read_env_file",
    "remove_provider",
    "resolve_env_file",
    "save_provider",
    "update_bfts_models",
    "update_env_file",
    "validate_provider_model",
    "workspace_config_path",
]
