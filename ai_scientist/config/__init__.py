"""
AI Scientist 配置模块
"""

from .paths import (
    OUTPUT_DIR,
    OUTPUT_PATH,
    CACHE_DIR,
    IDEAS_DIR,
    EXPERIMENTS_DIR,
    PROJECTS_DIR,
    ensure_output_dirs,
    get_experiment_dir,
    get_idea_path,
    get_project_dir,
    resolve_output_dir_value,
    resolve_output_path,
)
from .venues import (
    CONFERENCE_TARGET_VENUES,
    DEFAULT_TARGET_VENUE,
    TARGET_VENUES,
    TARGET_VENUE_SET,
)

__all__ = [
    "OUTPUT_DIR",
    "OUTPUT_PATH",
    "CACHE_DIR",
    "IDEAS_DIR",
    "EXPERIMENTS_DIR",
    "PROJECTS_DIR",
    "ensure_output_dirs",
    "get_experiment_dir",
    "get_idea_path",
    "get_project_dir",
    "resolve_output_dir_value",
    "resolve_output_path",
    "CONFERENCE_TARGET_VENUES",
    "DEFAULT_TARGET_VENUE",
    "TARGET_VENUES",
    "TARGET_VENUE_SET",
]
