from __future__ import annotations

from typing import Callable

from ai_scientist.utils.workflow_cli import (
    normalize_launcher_workflow_args,
)


def normalize_common_launcher_args(
    args,
    *,
    invalid_profile_logger: Callable[[ValueError], None] | None = None,
):
    return normalize_launcher_workflow_args(
        args,
        invalid_profile_logger=invalid_profile_logger,
    )
