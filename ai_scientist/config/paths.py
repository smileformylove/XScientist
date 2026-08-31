"""
XScientist output path configuration.

Directory layout:
<output_root>/
├── cache/
├── ideas/
├── experiments/
├── papers/
│   └── paper_YYYYMMDD_HHMMSS_idea_name/
│       ├── idea.json
│       ├── idea.md
│       ├── experiment/
│       ├── latex/
│       ├── paper.pdf
│       └── reviews/
└── batches/
    └── batch_YYYYMMDD_HHMMSS/
        ├── progress.json
        └── final_report.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Optional

_WINDOWS_RESERVED_COMPONENTS = {
    "aux",
    "con",
    "nul",
    "prn",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
_WINDOWS_FORBIDDEN_CHARS = frozenset('<>:"/\\|?*')


def is_windows_reserved_component(value: object) -> bool:
    """Return whether one filename component is a Windows device name."""

    normalized = unicodedata.normalize("NFKC", str(value or "")).strip().lower()
    return normalized.rstrip(" .").partition(".")[0] in _WINDOWS_RESERVED_COMPONENTS


def is_windows_unsafe_component(value: object) -> bool:
    """Return whether one component aliases or is invalid on normal Win32 paths."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    return bool(
        not text
        or text in {".", ".."}
        or text.endswith((" ", "."))
        or any(ord(character) < 32 for character in text)
        or any(character in _WINDOWS_FORBIDDEN_CHARS for character in text)
        or is_windows_reserved_component(text)
    )


# Repository root.
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# Unified output directory. Prefer RESEARCH_OUTPUT_DIR and keep
# AI_SCIENTIST_OUTPUT_DIR as a compatibility alias.
PRIMARY_OUTPUT_ENV_VAR = "RESEARCH_OUTPUT_DIR"
LEGACY_OUTPUT_ENV_VAR = "AI_SCIENTIST_OUTPUT_DIR"


def resolve_output_dir_value() -> str:
    return (
        os.environ.get(PRIMARY_OUTPUT_ENV_VAR)
        or os.environ.get(LEGACY_OUTPUT_ENV_VAR)
        or str(DEFAULT_RESEARCH_DIR)
    )


def resolve_output_path() -> Path:
    return Path(resolve_output_dir_value()).expanduser()


def _resolve_default_research_dir(
    *,
    home_dir: Path,
    platform_name: str,
    xdg_data_home: Optional[str],
    legacy_documents_dir_exists: bool,
    project_root: Optional[Path] = None,
    prefer_project_sibling: bool = True,
    project_parent_writable: Optional[bool] = None,
) -> Path:
    if prefer_project_sibling:
        resolved_project_root = (project_root or PROJECT_ROOT).expanduser().resolve()
        sibling_output_dir = (
            resolved_project_root.parent / f"{resolved_project_root.name}_outputs"
        )
        if project_parent_writable is None:
            project_parent_writable = os.access(sibling_output_dir.parent, os.W_OK)
        if project_parent_writable:
            return sibling_output_dir
    legacy_documents_dir = home_dir / "Documents" / "research"
    if legacy_documents_dir_exists:
        return legacy_documents_dir
    if platform_name == "nt":
        return home_dir / "AppData" / "Local" / "ai_scientist" / "research"
    base_data_dir = (
        Path(xdg_data_home).expanduser()
        if xdg_data_home
        else home_dir / ".local" / "share"
    )
    return base_data_dir / "ai_scientist" / "research"


def _default_research_dir() -> Path:
    home_dir = Path.home()
    legacy_documents_dir = home_dir / "Documents" / "research"
    return _resolve_default_research_dir(
        home_dir=home_dir,
        platform_name=os.name,
        xdg_data_home=os.environ.get("XDG_DATA_HOME"),
        legacy_documents_dir_exists=legacy_documents_dir.exists(),
        prefer_project_sibling=(PROJECT_ROOT / "pyproject.toml").is_file(),
    )


DEFAULT_RESEARCH_DIR = _default_research_dir()
# Backward-compatible import-time snapshots. Prefer resolve_output_path() in runtime-sensitive flows.
OUTPUT_DIR = resolve_output_dir_value()
OUTPUT_PATH = resolve_output_path()

# Common output paths.
CACHE_DIR = OUTPUT_PATH / "cache"
IDEAS_DIR = OUTPUT_PATH / "ideas"
EXPERIMENTS_DIR = OUTPUT_PATH / "experiments"
PROJECTS_DIR = OUTPUT_PATH / "projects"
PAPERS_DIR = OUTPUT_PATH / "papers"
BATCHES_DIR = OUTPUT_PATH / "batches"


def _resolve_output_root(output_root: str | Path | None = None) -> Path:
    if output_root is None:
        return resolve_output_path()
    return Path(output_root).expanduser()


def safe_path_component(
    value: object,
    *,
    fallback: str = "item",
    max_length: int = 64,
) -> str:
    """Return one portable filename component derived from untrusted text.

    Idea names can originate in model output or caller-supplied JSON.  They are
    labels, never paths: separators, traversal tokens, control characters, and
    platform-specific punctuation must not affect where artifacts are stored.
    """

    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    raw_value = unicodedata.normalize("NFKC", str(value or "")).strip()
    normalized = raw_value.lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "-", normalized).strip("-._")
    normalized = re.sub(r"-+", "-", normalized)
    fallback_value = (
        re.sub(r"[^a-z0-9._-]+", "-", str(fallback or "item").lower()).strip("-._")
        or "item"
    )
    if not normalized and raw_value:
        digest = hashlib.sha256(raw_value.encode("utf-8")).hexdigest()[:8]
        normalized = f"{fallback_value}-{digest}"
    if is_windows_reserved_component(normalized):
        normalized = f"{fallback_value}-{normalized}"
    return (normalized or fallback_value)[:max_length].rstrip("-._") or "item"


def _keyed_artifact_component(
    value: object,
    *,
    fallback: str,
    max_length: int,
    legacy_value: object | None = None,
    force_keyed: bool = False,
) -> str:
    """Preserve safe legacy names and key every lossy transformation.

    ``safe_path_component`` is intentionally a human-readable slug, so distinct
    inputs may share it. Artifact helpers need stronger semantics: a label may
    keep its historical spelling only when that spelling is already one
    portable component. Any normalization, truncation, or unsafe character adds
    a short digest of the original value.
    """

    raw = str(value or "")
    legacy = raw if legacy_value is None else str(legacy_value or "")
    legacy_is_lossless = (
        not force_keyed
        and raw == unicodedata.normalize("NFKC", raw)
        and raw == legacy
        and not is_windows_unsafe_component(legacy)
        and len(legacy) <= max_length
        and len(legacy.encode("utf-8")) <= max_length
    )
    if legacy_is_lossless:
        return legacy
    digest = content_identity(raw, length=8)
    slug_limit = max(8, max_length - len(digest) - 1)
    slug = safe_path_component(legacy, fallback=fallback, max_length=slug_limit)
    return f"{slug}-{digest}"


def content_identity(value: object, *, length: int = 12) -> str:
    """Return a stable short SHA-256 identity for JSON-compatible content."""

    if not 8 <= int(length) <= 64:
        raise ValueError("identity length must be between 8 and 64")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ValueError("value cannot be converted to a stable identity") from exc
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def idea_storage_key(
    idea: object,
    *,
    idea_index: int | None = None,
    max_slug_length: int = 48,
) -> str:
    """Build a collision-resistant, traversal-safe key for an idea artifact."""

    name = idea.get("Name") if isinstance(idea, dict) else idea
    slug = safe_path_component(name, fallback="idea", max_length=max_slug_length)
    digest = content_identity(idea)
    prefix = ""
    if idea_index is not None:
        if (
            isinstance(idea_index, bool)
            or not isinstance(idea_index, int)
            or idea_index < 0
        ):
            raise ValueError("idea_index must be a non-negative integer")
        prefix = f"{idea_index:04d}-"
    return f"{prefix}{slug}-{digest}"


def confined_path(root: str | Path, *components: str) -> Path:
    """Resolve a child path and fail if it escapes ``root`` for any reason."""

    # Keep the caller-visible spelling (notably macOS ``/var`` versus
    # ``/private/var``) while resolving symlinks only for the confinement check.
    lexical_root = Path(os.path.abspath(Path(root).expanduser()))
    candidate = Path(os.path.abspath(lexical_root.joinpath(*components)))
    resolved_root = lexical_root.resolve()
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("artifact path escapes its configured root") from exc
    return candidate


def ensure_output_dirs(output_root: str | Path | None = None):
    """Ensure the standard output directories exist."""
    root = _resolve_output_root(output_root)
    dirs = [
        root,
        root / "cache",
        root / "ideas",
        root / "experiments",
        root / "projects",
        root / "papers",
        root / "batches",
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_cache_env_vars(
    output_root: str | Path | None = None,
) -> dict[str, str]:
    """
    Return recommended cache-related environment variables for faster experiment/data runs.

    We avoid overriding user-provided env vars; use `apply_cache_env_vars()` for a safe setter.
    """
    cache_dir = _resolve_output_root(output_root) / "cache"
    hf_home = cache_dir / "huggingface"
    return {
        # HuggingFace hub + datasets caching
        "HF_HOME": str(hf_home),
        "HF_DATASETS_CACHE": str(hf_home / "datasets"),
        "HF_HUB_CACHE": str(hf_home / "hub"),
        # Backward/legacy knobs still honored by some stacks.
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        # PyTorch model/dataset cache (e.g., torchvision)
        "TORCH_HOME": str(cache_dir / "torch"),
        # Keep wandb runs out of repo/workspaces by default.
        "WANDB_DIR": str(cache_dir / "wandb"),
    }


def apply_cache_env_vars(
    *,
    override: bool = False,
    output_root: str | Path | None = None,
) -> dict[str, str]:
    """
    Apply the cache env vars to the current process, without clobbering existing values
    unless `override=True`.

    Returns the env vars that were set/ensured by this call.
    """
    env = get_cache_env_vars(output_root=output_root)
    applied: dict[str, str] = {}
    for key, value in env.items():
        if (not override) and os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
        try:
            Path(value).expanduser().mkdir(parents=True, exist_ok=True)
        except OSError:
            # Best-effort: cache dirs can still be created lazily by libraries.
            pass
    return applied


def get_experiment_dir(
    idea_name: str,
    attempt_id: int = 0,
    output_root: str | Path | None = None,
    *,
    idea_identity: object | None = None,
    idea_index: int | None = None,
) -> Path:
    """
    获取实验目录路径

    Args:
        idea_name: 想法名称
        attempt_id: 尝试ID

    Returns:
        实验目录的完整路径
    """
    if (
        isinstance(attempt_id, bool)
        or not isinstance(attempt_id, int)
        or attempt_id < 0
    ):
        raise ValueError("attempt_id must be a non-negative integer")
    root = _resolve_output_root(output_root)
    # Microseconds reduce accidental collisions across processes; the content
    # identity and optional index keep duplicate labels distinct and auditable.
    date = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    identity = idea_name if idea_identity is None else idea_identity
    storage_key = idea_storage_key(identity, idea_index=idea_index)
    return confined_path(
        root,
        "experiments",
        f"{date}_{storage_key}_attempt_{attempt_id}",
    )


def get_idea_path(base_name: str, output_root: str | Path | None = None) -> Path:
    """
    获取 idea 文件路径（统一存放在输出目录的 ideas/ 下）

    Args:
        base_name: idea文件的基础名称（不含扩展名）

    Returns:
        idea JSON文件的完整路径
    """
    root = _resolve_output_root(output_root)
    filename = _keyed_artifact_component(
        base_name,
        fallback="idea",
        max_length=96,
    )
    return confined_path(root, "ideas", f"{filename}.json")


def get_project_dir(project_name: str, output_root: str | Path | None = None) -> Path:
    """
    获取项目目录路径

    Args:
        project_name: 项目名称

    Returns:
        项目目录的完整路径
    """
    root = _resolve_output_root(output_root)
    projects_root = confined_path(root, "projects")
    project_dir = confined_path(projects_root, project_name)
    project_components = [
        component for component in re.split(r"[\\/]+", str(project_name)) if component
    ]
    if any(
        is_windows_reserved_component(component) for component in project_components
    ):
        raise ValueError("project name is reserved on Windows")
    if any(is_windows_unsafe_component(component) for component in project_components):
        raise ValueError("project name is not a portable directory component")
    return project_dir


def get_batch_dir(
    batch_name: str = None,
    output_root: str | Path | None = None,
) -> Path:
    """
    获取批次目录路径，用于连续产生论文

    Args:
        batch_name: 批次名称，如果为None则使用时间戳

    Returns:
        批次目录的完整路径
    """
    root = _resolve_output_root(output_root)
    if batch_name is None:
        batch_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_batch_name = _keyed_artifact_component(
        batch_name,
        fallback="batch",
        max_length=96,
    )
    return confined_path(root, "batches", f"batch_{safe_batch_name}")


def get_paper_dir(
    idea_name: str,
    paper_type: str = "icbinb",
    timestamp: str = None,
    output_root: str | Path | None = None,
) -> Path:
    """
    获取单篇论文的独立目录路径

    Args:
        idea_name: 想法名称
        paper_type: 论文类型 (icbinb, normal, journal, etc.)
        timestamp: 时间戳，如果为None则使用当前时间

    Returns:
        论文独立目录的完整路径
    """
    root = _resolve_output_root(output_root)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 论文目录命名格式: paper_时间戳_想法名_类型
    legacy_name = str(idea_name).replace(" ", "_").replace("/", "_").lower()
    safe_name = _keyed_artifact_component(
        idea_name,
        legacy_value=legacy_name,
        fallback="idea",
        max_length=64,
    )
    safe_timestamp = _keyed_artifact_component(
        timestamp,
        fallback="timestamp",
        max_length=32,
        force_keyed=re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?", str(timestamp)) is None,
    )
    safe_type = _keyed_artifact_component(
        paper_type,
        fallback="paper",
        max_length=32,
        force_keyed=str(paper_type) not in PAPER_TYPES,
    )
    paper_dir_name = f"paper_{safe_timestamp}_{safe_name}_{safe_type}"
    paper_dir = confined_path(root, "papers", paper_dir_name)

    # Existing safe historical directories remain resumable. New unsafe or
    # ambiguous labels always use the keyed path above.
    legacy_dir_name = f"paper_{timestamp}_{legacy_name}_{paper_type}"
    try:
        legacy_dir = confined_path(root, "papers", legacy_dir_name)
    except ValueError:
        legacy_dir = None
    if legacy_dir is not None and legacy_dir.exists() and not paper_dir.exists():
        try:
            legacy_idea = json.loads(
                (legacy_dir / "idea.json").read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            legacy_idea = None
        if isinstance(legacy_idea, dict) and str(legacy_idea.get("Name") or "") == str(
            idea_name
        ):
            return legacy_dir
    return paper_dir


def create_paper_structure(paper_dir: Path) -> dict:
    """
    创建单篇论文的目录结构

    Args:
        paper_dir: 论文目录路径

    Returns:
        创建的目录字典
    """
    dirs = {
        "root": paper_dir,
        "experiment": paper_dir / "experiment",
        "latex": paper_dir / "latex",
        "reviews": paper_dir / "reviews",
        "logs": paper_dir / "logs",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


# 论文类型配置
PAPER_TYPES = {
    "icbinb": {
        "name": "ICLR Workshop (ICBINB)",
        "page_limit": 4,
        "template": "blank_icbinb_latex",
        "description": "4页 workshop 论文",
    },
    "normal": {
        "name": "Standard Conference Paper",
        "page_limit": 8,
        "template": "blank_icml_latex",
        "description": "8页标准会议论文",
    },
    "journal": {
        "name": "Journal Paper",
        "page_limit": 12,
        "template": "blank_icml_latex",
        "description": "12页期刊论文",
    },
    "extended": {
        "name": "Extended Abstract",
        "page_limit": 2,
        "template": "blank_icbinb_latex",
        "description": "2页扩展摘要",
    },
}


def get_paper_type_config(paper_type: str) -> dict:
    """
    获取论文类型配置

    Args:
        paper_type: 论文类型

    Returns:
        论文类型配置字典
    """
    return PAPER_TYPES.get(paper_type, PAPER_TYPES["icbinb"])


# 自动创建输出目录（当模块被导入时）
if __name__ != "__main__":
    try:
        ensure_output_dirs()
    except OSError:
        # Best-effort only: users can still override output root via env/CLI.
        pass
