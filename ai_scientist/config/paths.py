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

import os
from pathlib import Path
from datetime import datetime
from typing import Optional

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
    base_data_dir = Path(xdg_data_home).expanduser() if xdg_data_home else home_dir / ".local" / "share"
    return base_data_dir / "ai_scientist" / "research"


def _default_research_dir() -> Path:
    home_dir = Path.home()
    legacy_documents_dir = home_dir / "Documents" / "research"
    return _resolve_default_research_dir(
        home_dir=home_dir,
        platform_name=os.name,
        xdg_data_home=os.environ.get("XDG_DATA_HOME"),
        legacy_documents_dir_exists=legacy_documents_dir.exists(),
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
) -> Path:
    """
    获取实验目录路径

    Args:
        idea_name: 想法名称
        attempt_id: 尝试ID

    Returns:
        实验目录的完整路径
    """
    root = _resolve_output_root(output_root)
    date = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / "experiments" / f"{date}_{idea_name}_attempt_{attempt_id}"


def get_idea_path(base_name: str, output_root: str | Path | None = None) -> Path:
    """
    获取 idea 文件路径（统一存放在输出目录的 ideas/ 下）

    Args:
        base_name: idea文件的基础名称（不含扩展名）

    Returns:
        idea JSON文件的完整路径
    """
    root = _resolve_output_root(output_root)
    return root / "ideas" / f"{base_name}.json"


def get_project_dir(project_name: str, output_root: str | Path | None = None) -> Path:
    """
    获取项目目录路径

    Args:
        project_name: 项目名称

    Returns:
        项目目录的完整路径
    """
    root = _resolve_output_root(output_root)
    return root / "projects" / project_name


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
    return root / "batches" / f"batch_{batch_name}"


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
    safe_name = idea_name.replace(" ", "_").replace("/", "_").lower()
    paper_dir_name = f"paper_{timestamp}_{safe_name}_{paper_type}"

    return root / "papers" / paper_dir_name


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
        "description": "4页 workshop 论文"
    },
    "normal": {
        "name": "Standard Conference Paper",
        "page_limit": 8,
        "template": "blank_icml_latex",
        "description": "8页标准会议论文"
    },
    "journal": {
        "name": "Journal Paper",
        "page_limit": 12,
        "template": "blank_icml_latex",
        "description": "12页期刊论文"
    },
    "extended": {
        "name": "Extended Abstract",
        "page_limit": 2,
        "template": "blank_icbinb_latex",
        "description": "2页扩展摘要"
    }
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
