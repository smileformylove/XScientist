from __future__ import annotations

import contextvars
import itertools
import json
import os
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from ai_scientist.utils.atomic_io import atomic_write_bytes
from ai_scientist.utils.privacy import redact_sensitive_text
from ai_scientist.utils.provider_registry import resolve_model_provider
from .dependency_profiles import (
    installation_command,
    missing_provider_modules,
    provider_client_modules,
)

CONFIG_SCHEMA_VERSION = 1
CONFIG_RELATIVE_PATH = Path(".xscientist") / "providers.json"
DEFAULT_ENV_FILE = ".env"
DEFAULT_MODELS = {"zhipu": "glm-4-flash"}
PROVIDER_ALIASES = {"custom": "openai_compat"}
PLACEHOLDER_VALUES = {"", "replace-me", "your-api-key", "your_api_key_here"}

# ``load_workspace_environment`` necessarily exposes the selected workspace to
# legacy code through ``os.environ``.  Keep a small, in-memory ownership map so
# a later workspace load can distinguish values injected by XScientist from
# credentials explicitly supplied by the user's shell.  This prevents a
# process that services multiple workspaces (the desktop app and test runners
# do this routinely) from silently routing workspace B through workspace A's
# endpoint or account.
_MANAGED_ENV_VALUES: dict[str, tuple[str, str]] = {}

# Process environment updates are shared by every in-process CLI invocation.
# A plain before/after snapshot cannot identify who wrote a value: a later
# snapshot can accidentally absorb another thread's update, and value-only
# compare-and-swap cannot detect an ABA write of the same value.  Keep each
# controlled mutation atomic and give it a unique provenance token instead.
_ENVIRONMENT_LOCK = threading.RLock()
_ENVIRONMENT_GENERATIONS: dict[str, int] = {}
_ENVIRONMENT_GENERATION_COUNTER = itertools.count(1)
_MISSING_ENVIRONMENT_VALUE = object()


@dataclass(frozen=True)
class _ManagedEnvironmentMutation:
    name: str
    before_process: object
    before_managed: object
    before_generation: object
    after_process: object
    after_managed: object
    after_generation: int


@dataclass
class ManagedEnvironmentTransaction:
    """Journal only environment writes made by the current execution context."""

    _mutations: list[_ManagedEnvironmentMutation] = field(default_factory=list)
    _active: bool = True
    _context_token: contextvars.Token["ManagedEnvironmentTransaction | None"] | None = (
        None
    )
    _conflicts: tuple[str, ...] = ()

    def _record(self, mutation: _ManagedEnvironmentMutation) -> None:
        if not self._active:  # pragma: no cover - internal invariant guard
            raise RuntimeError("managed environment transaction is closed")
        self._mutations.append(mutation)

    def _leave_context(self) -> None:
        token = self._context_token
        if token is None:
            return
        self._context_token = None
        if _ACTIVE_ENVIRONMENT_TRANSACTION.get() is self:
            _ACTIVE_ENVIRONMENT_TRANSACTION.reset(token)

    def commit(self) -> None:
        """Keep the journaled state and stop attributing later writes to it."""

        with _ENVIRONMENT_LOCK:
            self._active = False
        self._leave_context()

    def rollback(self) -> tuple[str, ...]:
        """Undo still-owned mutations while preserving every concurrent write."""

        if not self._active:
            return self._conflicts
        conflicts: set[str] = set()
        missing = _MISSING_ENVIRONMENT_VALUE
        with _ENVIRONMENT_LOCK:
            for mutation in reversed(self._mutations):
                name = mutation.name
                current_process = os.environ.get(name, missing)
                current_managed = _MANAGED_ENV_VALUES.get(name, missing)
                current_generation = _ENVIRONMENT_GENERATIONS.get(name, missing)
                if not (
                    _environment_values_equal(current_process, mutation.after_process)
                    and _environment_values_equal(
                        current_managed, mutation.after_managed
                    )
                    and _environment_values_equal(
                        current_generation, mutation.after_generation
                    )
                ):
                    conflicts.add(name)
                    continue
                _restore_process_environment(name, mutation.before_process)
                _restore_managed_environment(name, mutation.before_managed)
                if mutation.before_generation is missing:
                    _ENVIRONMENT_GENERATIONS.pop(name, None)
                else:
                    _ENVIRONMENT_GENERATIONS[name] = int(mutation.before_generation)
            self._active = False
            self._conflicts = tuple(sorted(conflicts))
        self._leave_context()
        return self._conflicts


_ACTIVE_ENVIRONMENT_TRANSACTION: contextvars.ContextVar[
    ManagedEnvironmentTransaction | None
] = contextvars.ContextVar("xscientist_managed_environment_transaction", default=None)


def _environment_values_equal(left: object, right: object) -> bool:
    missing = _MISSING_ENVIRONMENT_VALUE
    if left is missing or right is missing:
        return left is right
    return left == right


def _restore_process_environment(name: str, value: object) -> None:
    if value is _MISSING_ENVIRONMENT_VALUE:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(value)


def _restore_managed_environment(name: str, value: object) -> None:
    if value is _MISSING_ENVIRONMENT_VALUE:
        _MANAGED_ENV_VALUES.pop(name, None)
    else:
        owner, managed_value = value  # type: ignore[misc]
        _MANAGED_ENV_VALUES[name] = (str(owner), str(managed_value))


def begin_managed_environment_transaction() -> ManagedEnvironmentTransaction:
    """Attribute controlled mutations in this context to a new transaction."""

    if _ACTIVE_ENVIRONMENT_TRANSACTION.get() is not None:
        raise RuntimeError("nested managed environment transactions are unsupported")
    transaction = ManagedEnvironmentTransaction()
    transaction._context_token = _ACTIVE_ENVIRONMENT_TRANSACTION.set(transaction)
    return transaction


def _replace_managed_environment(
    name: str,
    *,
    process_value: str | None,
    managed_value: tuple[str, str] | None,
) -> None:
    """Atomically replace one process value and its ownership attribution."""

    name = str(name)
    missing = _MISSING_ENVIRONMENT_VALUE
    after_process: object = missing if process_value is None else str(process_value)
    after_managed: object = missing if managed_value is None else managed_value
    with _ENVIRONMENT_LOCK:
        before_process = os.environ.get(name, missing)
        before_managed = _MANAGED_ENV_VALUES.get(name, missing)
        before_generation = _ENVIRONMENT_GENERATIONS.get(name, missing)
        if _environment_values_equal(
            before_process, after_process
        ) and _environment_values_equal(before_managed, after_managed):
            return
        _restore_process_environment(name, after_process)
        _restore_managed_environment(name, after_managed)
        generation = next(_ENVIRONMENT_GENERATION_COUNTER)
        _ENVIRONMENT_GENERATIONS[name] = generation
        transaction = _ACTIVE_ENVIRONMENT_TRANSACTION.get()
        if transaction is not None:
            transaction._record(
                _ManagedEnvironmentMutation(
                    name=name,
                    before_process=before_process,
                    before_managed=before_managed,
                    before_generation=before_generation,
                    after_process=after_process,
                    after_managed=after_managed,
                    after_generation=generation,
                )
            )


@dataclass(frozen=True)
class _ProviderFileState:
    kind: str
    content: bytes | None = None
    mode: int | None = None
    device: int | None = None
    inode: int | None = None


@dataclass(frozen=True)
class _ProviderFileMutation:
    path: Path
    before: _ProviderFileState
    before_generation: object
    after: _ProviderFileState
    after_generation: int


@dataclass
class ProviderFileTransaction:
    """CAS rollback for the exact provider files written by one command."""

    _mutations: list[_ProviderFileMutation] = field(default_factory=list)
    _active: bool = True
    _context_token: contextvars.Token["ProviderFileTransaction | None"] | None = None
    _conflicts: tuple[str, ...] = ()

    def _record(self, mutation: _ProviderFileMutation) -> None:
        if any(existing.path == mutation.path for existing in self._mutations):
            raise RuntimeError("provider file transaction wrote one path twice")
        self._mutations.append(mutation)

    def _leave_context(self) -> None:
        token = self._context_token
        if token is None:
            return
        self._context_token = None
        if _ACTIVE_PROVIDER_FILE_TRANSACTION.get() is self:
            _ACTIVE_PROVIDER_FILE_TRANSACTION.reset(token)

    def commit(self) -> None:
        with _PROVIDER_FILE_LOCK:
            self._active = False
        self._leave_context()

    def rollback(self) -> tuple[str, ...]:
        if not self._active:
            return self._conflicts
        missing = _MISSING_ENVIRONMENT_VALUE
        conflicts: set[str] = set()
        with _PROVIDER_FILE_LOCK:
            for mutation in reversed(self._mutations):
                try:
                    current = _capture_provider_file_state(mutation.path)
                except (OSError, ProviderConfigError):
                    conflicts.add(str(mutation.path))
                    continue
                generation = _PROVIDER_FILE_GENERATIONS.get(mutation.path, missing)
                if current != mutation.after or not _environment_values_equal(
                    generation, mutation.after_generation
                ):
                    conflicts.add(str(mutation.path))
                    continue
                try:
                    _restore_provider_file_state(mutation.path, mutation.before)
                except (OSError, ProviderConfigError):
                    conflicts.add(str(mutation.path))
                    continue
                if mutation.before_generation is missing:
                    _PROVIDER_FILE_GENERATIONS.pop(mutation.path, None)
                else:
                    _PROVIDER_FILE_GENERATIONS[mutation.path] = int(
                        mutation.before_generation
                    )
            self._active = False
            self._conflicts = tuple(sorted(conflicts))
        self._leave_context()
        return self._conflicts


_PROVIDER_FILE_LOCK = threading.RLock()
_PROVIDER_FILE_GENERATIONS: dict[Path, int] = {}
_PROVIDER_FILE_GENERATION_COUNTER = itertools.count(1)
_ACTIVE_PROVIDER_FILE_TRANSACTION: contextvars.ContextVar[
    ProviderFileTransaction | None
] = contextvars.ContextVar("xscientist_provider_file_transaction", default=None)


def begin_provider_file_transaction() -> ProviderFileTransaction:
    if _ACTIVE_PROVIDER_FILE_TRANSACTION.get() is not None:
        raise RuntimeError("nested provider file transactions are unsupported")
    transaction = ProviderFileTransaction()
    transaction._context_token = _ACTIVE_PROVIDER_FILE_TRANSACTION.set(transaction)
    return transaction


def _capture_provider_file_state(path: Path) -> _ProviderFileState:
    target = Path(path).expanduser().absolute()
    try:
        stat_result = target.lstat()
    except FileNotFoundError:
        return _ProviderFileState(kind="absent")
    if target.is_symlink() or not target.is_file():
        return _ProviderFileState(
            kind="unsafe",
            mode=stat_result.st_mode & 0o7777,
            device=stat_result.st_dev,
            inode=stat_result.st_ino,
        )
    content = target.read_bytes()
    confirmed = target.lstat()
    if (confirmed.st_dev, confirmed.st_ino) != (
        stat_result.st_dev,
        stat_result.st_ino,
    ):
        raise ProviderConfigError("provider file changed while it was inspected")
    return _ProviderFileState(
        kind="file",
        content=content,
        mode=confirmed.st_mode & 0o7777,
        device=confirmed.st_dev,
        inode=confirmed.st_ino,
    )


def _restore_provider_file_state(path: Path, state: _ProviderFileState) -> None:
    if state.kind == "absent":
        path.unlink(missing_ok=True)
        return
    if state.kind != "file" or state.content is None or state.mode is None:
        raise ProviderConfigError("provider file rollback snapshot is unsafe")
    atomic_write_bytes(path, state.content)
    path.chmod(state.mode)


def _atomic_provider_text_replace(
    path: Path,
    content: str,
    *,
    mode: int,
) -> tuple[int, int]:
    """Replace a file and return the exact identity of our temporary inode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temp = Path(raw_temp)
    try:
        try:
            os.fchmod(descriptor, mode)
        except (AttributeError, OSError):
            pass
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            stat_result = os.fstat(handle.fileno())
        temp.replace(path)
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_descriptor = None
        if directory_descriptor is not None:
            try:
                os.fsync(directory_descriptor)
            except OSError:
                pass
            finally:
                os.close(directory_descriptor)
        return stat_result.st_dev, stat_result.st_ino
    finally:
        temp.unlink(missing_ok=True)


def _tracked_provider_file_write(
    path: Path,
    content: str,
    *,
    secure: bool = False,
) -> None:
    target = Path(path).expanduser().absolute()
    encoded = content.encode("utf-8")
    missing = _MISSING_ENVIRONMENT_VALUE
    with _PROVIDER_FILE_LOCK:
        before = _capture_provider_file_state(target)
        if before.kind not in {"absent", "file"}:
            raise ProviderConfigError("provider file must be a regular file")
        before_generation = _PROVIDER_FILE_GENERATIONS.get(target, missing)
        expected_identity = _atomic_provider_text_replace(
            target,
            content,
            mode=(0o600 if secure else int(before.mode or 0o600)),
        )
        after = _capture_provider_file_state(target)
        if (
            after.kind != "file"
            or after.content != encoded
            or (after.device, after.inode) != expected_identity
        ):
            raise ProviderConfigError(
                "provider file changed concurrently after atomic replacement"
            )
        generation = next(_PROVIDER_FILE_GENERATION_COUNTER)
        _PROVIDER_FILE_GENERATIONS[target] = generation
        transaction = _ACTIVE_PROVIDER_FILE_TRANSACTION.get()
        if transaction is not None:
            transaction._record(
                _ProviderFileMutation(
                    path=target,
                    before=before,
                    before_generation=before_generation,
                    after=after,
                    after_generation=generation,
                )
            )


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
        ProviderField("OPENAI_COMPAT_API_KEY"),
        ProviderField("OPENAI_COMPAT_BASE_URL", secret=False),
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


def normalize_provider_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    return PROVIDER_ALIASES.get(normalized, normalized)


def normalize_provider_model(provider: str, model: str | None) -> str:
    """Make a user-entered model ID unambiguous for the selected provider.

    Provider prefixes remain part of the persisted contract, but people should
    not need to know that internal routing rule during first-run setup.  A bare
    model selected for ``openai_compat`` is always qualified because that
    explicit provider choice must win even when the model name is also known
    by a first-party route.  Other ambiguous bare IDs receive the selected
    provider prefix.
    """

    normalized = normalize_provider_name(provider)
    selected = str(model or "").strip()
    if not selected or normalized not in PROVIDER_FIELDS:
        return selected
    if selected.startswith("custom/") and normalized == "openai_compat":
        selected = "openai_compat/" + selected.split("/", 1)[1]
    # A slash is explicit routing metadata, not a friendly bare model name.
    # Never rewrite it: a mismatched or malformed prefix must keep failing
    # closed when provider configuration is loaded.
    if "/" in selected:
        return selected
    if normalized == "openai_compat":
        return f"openai_compat/{selected}"
    try:
        resolved_provider = resolve_model_provider(selected).provider
        if resolved_provider == normalized:
            return selected
    except ValueError:
        resolved_provider = None
    # The registry deliberately routes unknown bare IDs through OpenAI as its
    # historical fallback.  In that ambiguous case, the explicit provider
    # selected by the user is the stronger signal.  Do not rewrite IDs such as
    # ``glm-4-flash`` that already resolve to a different concrete provider.
    can_disambiguate = resolved_provider == "openai" and normalized != "openai"
    can_route_anthropic_cloud = resolved_provider == "anthropic" and normalized in {
        "bedrock",
        "vertex_ai",
    }
    if not (can_disambiguate or can_route_anthropic_cloud):
        return selected
    prefixed = f"{normalized}/{selected}"
    try:
        if resolve_model_provider(prefixed).provider == normalized:
            return prefixed
    except ValueError:
        pass
    return selected


def _ollama_base_url(environ: Mapping[str, str] | None = None) -> str:
    source = os.environ if environ is None else environ
    base_url = str(
        source.get("OLLAMA_BASE_URL")
        or source.get("OLLAMA_HOST")
        or "http://localhost:11434/v1"
    ).strip()
    if not base_url:
        return ""
    if "://" not in base_url:
        base_url = "http://" + base_url
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return base_url


def _ollama_models(
    *,
    timeout: float,
    environ: Mapping[str, str] | None,
) -> tuple[list[str], str | None]:
    base_url = _ollama_base_url(environ)
    if not base_url:
        return [], "Ollama endpoint is not configured"
    request = urllib.request.Request(
        base_url + "/api/tags",
        headers={"Accept": "application/json", "User-Agent": "xscientist-local"},
    )
    parsed_url = urllib.parse.urlparse(base_url)
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
        else urllib.request.build_opener()
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return [], "Ollama is not reachable; start it with `ollama serve`"
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return [], "Ollama returned an invalid model-list response"
    names = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("model") or "").strip()
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        families = {
            str(item).strip().lower()
            for item in (details.get("families") or [])
            if str(item).strip()
        }
        family = str(details.get("family") or "").strip().lower()
        if "embed" in name.lower() or family == "bert" or "bert" in families:
            continue
        if name:
            names.append(normalize_provider_model("ollama", name))
    return list(dict.fromkeys(names)), None


def discover_provider_models(
    provider: str,
    *,
    timeout: float = 0.75,
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Return locally discoverable model IDs without contacting paid APIs."""

    normalized = str(provider or "").strip().lower()
    if normalized != "ollama":
        return []
    models, _error = _ollama_models(timeout=timeout, environ=environ)
    return models


def probe_provider_model(
    provider: str,
    model: str | None,
    *,
    timeout: float = 0.75,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify free local providers without making a paid API request."""

    normalized = str(provider or "").strip().lower()
    if normalized != "ollama":
        return {
            "checked": False,
            "ok": True,
            "service_reachable": None,
            "model_available": None,
            "models": [],
            "error": None,
        }
    models, error = _ollama_models(timeout=timeout, environ=environ)
    selected = normalize_provider_model(normalized, model)
    service_reachable = error is None
    model_available = (
        bool(selected and selected in models) if service_reachable else False
    )
    if service_reachable and not selected:
        error = "No Ollama model is selected"
    elif service_reachable and not model_available:
        model_name = selected.split("/", 1)[-1] if selected else "<model>"
        error = (
            f"Ollama model {model_name!r} is not installed; "
            f"run `ollama pull {model_name}`"
        )
    return {
        "checked": True,
        "ok": bool(service_reachable and model_available),
        "service_reachable": service_reachable,
        "model_available": model_available,
        "models": models,
        "error": error,
    }


def workspace_config_path(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / CONFIG_RELATIVE_PATH


def _normalized_private_env_name(env_file: str) -> str:
    """Limit credential storage to an ignored, dedicated root-level env file."""

    relative = Path(str(env_file or DEFAULT_ENV_FILE).strip())
    name = relative.as_posix()
    if relative.is_absolute():
        raise ProviderConfigError("provider env_file must be relative to the workspace")
    if ".." in relative.parts:
        raise ProviderConfigError("provider env_file cannot escape the workspace")
    if len(relative.parts) != 1:
        raise ProviderConfigError(
            "provider env_file must be a root-level private .env file"
        )
    if not (name == ".env" or name.startswith(".env.")) or name == ".env.example":
        raise ProviderConfigError(
            "provider env_file must be .env or a private .env.* variant"
        )
    return name


def resolve_env_file(root: str | Path, env_file: str) -> Path:
    workspace = Path(root).expanduser().resolve()
    relative = Path(_normalized_private_env_name(env_file))
    target = (workspace / relative).resolve()
    try:
        target.relative_to(workspace)
    except ValueError as exc:
        raise ProviderConfigError(
            "provider env_file cannot escape the workspace"
        ) from exc
    unresolved = workspace / relative
    if unresolved.is_symlink():
        raise ProviderConfigError("provider env_file must not be a symlink")
    if unresolved.exists() and not unresolved.is_file():
        raise ProviderConfigError("provider env_file must be a regular file")
    return target


def discover_workspace_root() -> Path | None:
    explicit = str(os.environ.get("XSCIENTIST_WORKSPACE") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        return candidate if workspace_config_path(candidate).is_file() else None
    candidate = Path.cwd().resolve()
    for directory in (candidate, *candidate.parents):
        if workspace_config_path(directory).is_file():
            return directory
    return None


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
        raise ProviderConfigError(
            f"provider configuration not found: {CONFIG_RELATIVE_PATH.as_posix()}"
        )
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
    normalized_env_file = _normalized_private_env_name(env_file)
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
    for label, raw_value in (("provider", provider), ("model", model)):
        value = str(raw_value or "").strip()
        if value and redact_sensitive_text(value) != value:
            raise ProviderConfigError(
                f"{label} contains a credential or private literal; refusing "
                "to persist it"
            )
    normalized = normalize_provider_name(provider)
    if normalized not in PROVIDER_FIELDS:
        choices = ", ".join(PROVIDER_NAMES)
        raise ProviderConfigError(
            f"unknown provider {provider!r}; expected one of: {choices}"
        )
    selected = normalize_provider_model(
        normalized, model or DEFAULT_MODELS.get(normalized)
    )
    if not selected:
        raise ProviderConfigError(
            f"--model is required for provider {normalized!r}; use a provider-prefixed "
            f"ID such as {normalized}/<model>"
        )
    if redact_sensitive_text(selected) != selected:
        raise ProviderConfigError(
            "model contains a credential or private literal; refusing to persist it"
        )
    spec = resolve_model_provider(selected)
    if spec.provider != normalized:
        raise ProviderConfigError(
            f"model {selected!r} resolves to provider {spec.provider!r}, not "
            f"{normalized!r}; use a provider-prefixed ID such as "
            f"{normalized}/<model>"
        )
    return normalized, selected


def validate_custom_base_url(value: str) -> str:
    """Validate a user-selected OpenAI-compatible API root."""

    selected = str(value or "").strip().rstrip("/")
    if not selected:
        raise ProviderConfigError("--base-url cannot be empty")
    parsed = urllib.parse.urlsplit(selected)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ProviderConfigError("--base-url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ProviderConfigError("--base-url cannot contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ProviderConfigError("--base-url cannot contain a query or fragment")
    try:
        parsed.port
    except ValueError as exc:
        raise ProviderConfigError("--base-url contains an invalid port") from exc
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and parsed.hostname not in local_hosts:
        raise ProviderConfigError(
            "remote custom providers require HTTPS; HTTP is allowed only for loopback"
        )
    return selected


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


def _workspace_key(root: str | Path) -> str:
    return str(Path(root).expanduser().resolve())


def _process_environment_for_workspace(
    workspace: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return process values after removing stale managed values.

    An explicitly supplied mapping is treated as authoritative and is never
    mutated.  With the real process environment, only an exact value that was
    previously injected by XScientist for another workspace is removed; a
    user's unrelated shell variable therefore keeps its normal precedence.
    """

    if environ is not None:
        return dict(environ)
    with _ENVIRONMENT_LOCK:
        source = dict(os.environ)
        current = _workspace_key(workspace)
        for name, (owner, value) in list(_MANAGED_ENV_VALUES.items()):
            if owner != current and source.get(name) == value:
                source.pop(name, None)
        return source


def _effective_workspace_environment(
    workspace: str | Path,
    stored: Mapping[str, str],
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    # Explicit process/CLI values win over the current workspace file.  Stale
    # values injected by an earlier workspace have already been removed above.
    return {**dict(stored), **_process_environment_for_workspace(workspace, environ)}


def mark_managed_environment(
    root: str | Path,
    name: str,
    value: str,
) -> None:
    """Set and attribute a process variable to a workspace-owned value."""

    cleaned = str(value or "").strip()
    if not cleaned:
        return
    _replace_managed_environment(
        str(name),
        process_value=cleaned,
        managed_value=(_workspace_key(root), cleaned),
    )


def workspace_environment(
    root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Resolve a secret-bearing environment for one workspace without leaks.

    The returned mapping is safe to pass to provider clients.  It contains the
    workspace env file plus explicit process values, while filtering values
    that XScientist previously injected for a different workspace.
    """

    workspace = Path(root).expanduser().resolve()
    config = load_provider_config(workspace, missing_ok=True)
    env_file = resolve_env_file(
        workspace, str(config.get("env_file") or DEFAULT_ENV_FILE)
    )
    if env_file.is_file() and not _env_file_is_private(env_file):
        raise ProviderConfigError(
            "refusing to read credentials with broad permissions: "
            f"{env_file.relative_to(workspace).as_posix()}"
        )
    stored = read_env_file(env_file) if env_file.is_file() else {}
    return _effective_workspace_environment(workspace, stored, environ)


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
            "workspace": ".",
            "loaded_names": [],
            "error": (
                "refusing to load credentials with broad permissions: "
                f"{env_file.relative_to(workspace).as_posix()}"
            ),
        }
    stored = read_env_file(env_file)
    current_workspace_key = _workspace_key(workspace)
    loaded_names: list[str] = []
    overridden_names: list[str] = []
    # Selection, stale-value removal, and replacement form one short critical
    # section.  This prevents another workspace load from observing an env/map
    # pair half-written by this one.  The lock is re-entrant because the helpers
    # below also protect standalone callers.
    with _ENVIRONMENT_LOCK:
        for name, (owner, value) in list(_MANAGED_ENV_VALUES.items()):
            if owner != current_workspace_key and os.environ.get(name) == value:
                _replace_managed_environment(
                    name,
                    process_value=None,
                    managed_value=None,
                )
        process_environment = _process_environment_for_workspace(workspace)
        for name, value in stored.items():
            if not _is_configured_value(value):
                continue
            if name not in process_environment:
                mark_managed_environment(workspace, name, value)
                process_environment[name] = value
                loaded_names.append(name)
            elif process_environment[name] != value:
                # A value explicitly present in the shell remains authoritative;
                # report the distinction instead of silently claiming the file was
                # used.  Stale XScientist-managed values were removed above.
                owner_value = _MANAGED_ENV_VALUES.get(name)
                if owner_value and owner_value[0] == current_workspace_key:
                    mark_managed_environment(workspace, name, value)
                    process_environment[name] = value
                    overridden_names.append(name)
        active = str(config.get("active_provider") or "").strip()
        entry = config.get("providers", {}).get(active, {}) if active else {}
        model = str(entry.get("model") or "").strip() if isinstance(entry, dict) else ""
        if active:
            mark_managed_environment(workspace, "AI_SCIENTIST_ACTIVE_PROVIDER", active)
        if model:
            # The active workspace is an explicit local choice. It must replace a
            # stale process-wide default left by another workspace or desktop run.
            # Role-specific AI_SCIENTIST_MODEL_* values and CLI flags still win.
            mark_managed_environment(workspace, "AI_SCIENTIST_DEFAULT_MODEL", model)
            mark_managed_environment(workspace, "ZHIPU_DEFAULT_MODEL", model)
    return {
        "loaded": True,
        "workspace": ".",
        "active_provider": active or None,
        "model": model or None,
        "loaded_names": sorted(loaded_names),
        "overridden_names": sorted(overridden_names),
        "environment_scope": "workspace_plus_explicit_process",
    }


def _secure_atomic_write(path: Path, content: str) -> None:
    _tracked_provider_file_write(path, content, secure=True)


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


def _write_provider_config(workspace: Path, config: Mapping[str, Any]) -> None:
    """Match the generated canonical form to avoid formatting-only VCS dirt."""

    _tracked_provider_file_write(
        workspace_config_path(workspace),
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
    )


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
    _write_provider_config(workspace, config)
    return config


def activate_provider(root: str | Path, provider: str) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    normalized = normalize_provider_name(provider)
    config = load_provider_config(workspace, missing_ok=False)
    if normalized not in config["providers"]:
        raise ProviderConfigError(f"provider {normalized!r} is not configured")
    config["active_provider"] = normalized
    _write_provider_config(workspace, config)
    return config


def remove_provider(root: str | Path, provider: str) -> dict[str, Any]:
    workspace = Path(root).expanduser().resolve()
    normalized = normalize_provider_name(provider)
    config = load_provider_config(workspace, missing_ok=False)
    if normalized not in config["providers"]:
        raise ProviderConfigError(f"provider {normalized!r} is not configured")
    del config["providers"][normalized]
    if config.get("active_provider") == normalized:
        config["active_provider"] = next(iter(config["providers"]), None)
    _write_provider_config(workspace, config)
    return config


def update_bfts_models(path: str | Path, model: str) -> bool:
    target = Path(path)
    if not target.is_file():
        return False
    try:
        original = target.read_text(encoding="utf-8")
        payload = yaml.safe_load(original)
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderConfigError(f"cannot read BFTS config: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProviderConfigError("BFTS configuration must be a mapping")
    payload.setdefault("report", {})["model"] = model
    agent = payload.setdefault("agent", {})
    for key in ("code", "feedback", "vlm_feedback"):
        agent.setdefault(key, {})["model"] = model
    for key in ("summary", "select_node"):
        section = agent.get(key)
        if section is not None:
            if not isinstance(section, dict):
                raise ProviderConfigError(
                    f"BFTS configuration agent.{key} must be a mapping"
                )
            section["model"] = model
    generated_header = (
        original.splitlines()[0] + "\n"
        if original.startswith("# Generated by xscientist init ")
        else ""
    )
    _tracked_provider_file_write(
        target,
        generated_header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
    )
    return True


def provider_statuses(
    root: str | Path,
    *,
    find_spec: Any = None,
    probe_local: bool = False,
    environ: Mapping[str, str] | None = None,
    allow_uninitialized: bool = False,
) -> list[dict[str, Any]]:
    workspace = Path(root).expanduser().resolve()
    config = load_provider_config(workspace, missing_ok=allow_uninitialized)
    initialized = workspace_config_path(workspace).is_file()
    env_file = resolve_env_file(
        workspace, str(config.get("env_file") or DEFAULT_ENV_FILE)
    )
    env_error = (
        "credential file has broad permissions: "
        f"{env_file.relative_to(workspace).as_posix()}"
        if env_file.is_file() and not _env_file_is_private(env_file)
        else ""
    )
    # Never treat an arbitrary directory's .env as provider configuration.
    # Discovery-only listing is allowed to inspect process environment and
    # free local services, but local files require an initialized workspace.
    stored = read_env_file(env_file) if initialized else {}
    effective_environ = _effective_workspace_environment(workspace, stored, environ)
    active = str(config.get("active_provider") or "")
    configured_entries = config.get("providers", {})
    rows = []
    for provider in PROVIDER_NAMES:
        entry = configured_entries.get(provider, {})
        fields = PROVIDER_FIELDS[provider]
        missing_clients = missing_provider_modules(
            provider,
            **({"find_spec": find_spec} if find_spec is not None else {}),
        )
        missing = [
            " | ".join((field.name, *field.aliases))
            for field in fields
            if field.required
            and not _is_configured_value(
                configured_field_value(field, stored, effective_environ)
            )
        ]
        model = str(entry.get("model") or "") if isinstance(entry, dict) else ""
        local_probe = (
            probe_provider_model(
                provider,
                model,
                environ=effective_environ,
            )
            if probe_local
            and (provider == "ollama" or (isinstance(entry, dict) and bool(entry)))
            else {
                "checked": False,
                "ok": True,
                "service_reachable": None,
                "model_available": None,
                "models": [],
                "error": None,
            }
        )
        local_models = list(local_probe.get("models") or [])
        local_detected = bool(local_probe.get("service_reachable") and local_models)
        if local_detected and not model:
            local_probe = {
                **local_probe,
                "ok": True,
                "model_available": True,
                "error": None,
            }
        rows.append(
            {
                "provider": provider,
                "active": provider == active,
                "configured": isinstance(entry, dict) and bool(entry),
                "credentials_available": not missing,
                "client_available": not missing_clients,
                "client_modules": list(provider_client_modules(provider)),
                "missing_client_modules": missing_clients,
                "install_command": installation_command(provider),
                "ready": (
                    isinstance(entry, dict)
                    and bool(entry)
                    and not missing
                    and not missing_clients
                    and not env_error
                    and local_probe["ok"]
                ),
                "model": model,
                "suggested_model": model or (local_models[0] if local_models else ""),
                "local_detected": local_detected,
                "missing": missing,
                "error": env_error,
                "local_probe": local_probe,
                "environment_scope": (
                    "workspace_plus_explicit_process"
                    if environ is None
                    else "supplied_mapping"
                ),
            }
        )
    return rows


__all__ = [
    "CONFIG_RELATIVE_PATH",
    "DEFAULT_MODELS",
    "PROVIDER_ALIASES",
    "discover_provider_models",
    "empty_provider_config",
    "normalize_provider_name",
    "normalize_provider_model",
    "PROVIDER_FIELDS",
    "PROVIDER_NAMES",
    "ManagedEnvironmentTransaction",
    "ProviderFileTransaction",
    "ProviderConfigError",
    "ProviderField",
    "activate_provider",
    "begin_managed_environment_transaction",
    "begin_provider_file_transaction",
    "configured_field_value",
    "discover_workspace_root",
    "load_provider_config",
    "load_workspace_environment",
    "mark_managed_environment",
    "provider_config_payload",
    "probe_provider_model",
    "provider_statuses",
    "read_env_file",
    "remove_provider",
    "resolve_env_file",
    "save_provider",
    "update_bfts_models",
    "update_env_file",
    "validate_custom_base_url",
    "validate_provider_model",
    "workspace_config_path",
    "workspace_environment",
]
