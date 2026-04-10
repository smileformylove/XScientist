from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

from ai_scientist.config.paths import (
    PRIMARY_OUTPUT_ENV_VAR,
    apply_cache_env_vars,
    ensure_output_dirs,
    resolve_output_path,
)
from ai_scientist.writing_prompt_profiles import (
    DEFAULT_WRITING_PROFILE,
    normalize_writing_profile,
)
from ai_scientist.utils.provider_registry import missing_model_credentials

PROJECT_ROOT_ENV_VAR = "AI_SCIENTIST_ROOT"
WRITING_PROFILE_ENV_VAR = "AI_SCIENTIST_WRITING_PROFILE"


@dataclass(frozen=True)
class RuntimeContext:
    project_root: Path
    research_root: Path
    applied_cache_env: dict[str, str]


def resolve_entrypoint_project_root(source_file: str | Path) -> Path:
    return Path(source_file).expanduser().resolve().parent


def format_project_relative_path(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> str:
    target = Path(path).expanduser()
    base_root = (
        Path(project_root).expanduser()
        if project_root is not None
        else Path(
            os.environ.get(PROJECT_ROOT_ENV_VAR)
            or Path.cwd()
        ).expanduser()
    )
    try:
        return os.path.relpath(target, base_root)
    except ValueError:
        return str(target)


def resolve_writing_profile_env(
    *,
    invalid_profile_logger: Callable[[ValueError, Optional[str]], None] | None = None,
) -> str:
    raw = os.environ.get(WRITING_PROFILE_ENV_VAR)
    try:
        return normalize_writing_profile(raw)
    except ValueError as exc:
        if invalid_profile_logger is not None:
            invalid_profile_logger(exc, raw)
        return DEFAULT_WRITING_PROFILE


def require_env_var(
    name: str,
    *,
    missing_message: str,
    hint: str | None = None,
    logger: Callable[[str], None] = print,
) -> str:
    value = str(os.environ.get(name) or "").strip()
    if value:
        return value
    logger(missing_message)
    if hint:
        logger(hint)
    raise SystemExit(1)


def require_model_credentials(
    models: Iterable[str],
    *,
    logger: Callable[[str], None] = print,
) -> None:
    missing = missing_model_credentials(models, env=os.environ)
    if not missing:
        return

    logger("❌ 错误: 当前选择的模型缺少所需服务商凭证或兼容端点配置")
    for row in missing:
        logger(
            "  - 模型 "
            f"{row['model']} 需要 {row['display_name']} 凭证: {row['missing']}"
        )
    logger(
        "💡 提示: 你可以直接换成带 provider 前缀的模型，例如 "
        "`openai/gpt-4.1`、`gemini/gemini-2.5-pro-preview-03-25`、"
        "`openrouter/meta-llama/llama-3.1-405b-instruct`、"
        "`openai_compat/your-model`。"
    )
    raise SystemExit(1)


def initialize_runtime(
    *,
    source_file: str | Path,
    output_root: str | Path | None = None,
    set_project_root_env: bool = True,
    ensure_dirs: bool = True,
    apply_cache: bool = True,
) -> RuntimeContext:
    project_root = resolve_entrypoint_project_root(source_file)
    if set_project_root_env:
        os.environ[PROJECT_ROOT_ENV_VAR] = str(project_root)

    if output_root is None:
        research_root = resolve_output_path().resolve()
    else:
        research_root = Path(output_root).expanduser().resolve()
    os.environ[PRIMARY_OUTPUT_ENV_VAR] = str(research_root)

    if ensure_dirs:
        ensure_output_dirs(output_root=research_root)

    applied_cache_env = (
        apply_cache_env_vars(override=False, output_root=research_root)
        if apply_cache
        else {}
    )
    return RuntimeContext(
        project_root=project_root,
        research_root=research_root,
        applied_cache_env=applied_cache_env,
    )
