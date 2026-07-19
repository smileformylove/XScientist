from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ai_scientist.resources import resolve_bfts_config_path

from .models import CommandResult, ProjectRequest


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class XScientist:
    """High-level SDK for invoking isolated XScientist workflows."""

    def __init__(
        self,
        *,
        work_dir: str | Path | None = None,
        output_root: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.work_dir = (
            Path(work_dir).expanduser().resolve() if work_dir is not None else None
        )
        self.output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else None
        )
        self.env = {str(key): str(value) for key, value in (env or {}).items()}
        self.python_executable = str(python_executable or sys.executable)

    def project_command(self, request: ProjectRequest) -> list[str]:
        argv = request.to_argv()
        if request.output_root is None and self.output_root is not None:
            argv[1:1] = ["--output-root", str(self.output_root)]
        config_index = None
        for index, item in enumerate(argv):
            if item == "--bfts-config" and index + 1 < len(argv):
                config_index = index + 1
                break
        if config_index is None:
            argv.extend(
                ["--bfts-config", str(resolve_bfts_config_path(request.bfts_config))]
            )
        else:
            argv[config_index] = str(
                resolve_bfts_config_path(
                    argv[config_index],
                    base_dir=self.work_dir,
                )
            )
        return [self.python_executable, "-m", "ai_scientist.apps.project", *argv]

    def run_project(
        self,
        request: ProjectRequest,
        *,
        check: bool = False,
        timeout: float | None = None,
    ) -> CommandResult:
        return self.run_command(
            self.project_command(request), check=check, timeout=timeout
        )

    def run_command(
        self,
        command: Sequence[str],
        *,
        check: bool = False,
        timeout: float | None = None,
    ) -> CommandResult:
        env = os.environ.copy()
        env.update(self.env)
        if self.output_root is not None:
            env.setdefault("RESEARCH_OUTPUT_DIR", str(self.output_root))
        started_at = _now_iso()
        completed = subprocess.run(
            [str(item) for item in command],
            cwd=str(self.work_dir) if self.work_dir is not None else None,
            env=env,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(
            command=tuple(str(item) for item in command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            started_at=started_at,
            finished_at=_now_iso(),
        )
        if check and not result.ok:
            raise subprocess.CalledProcessError(
                result.returncode,
                list(result.command),
                output=result.stdout,
                stderr=result.stderr,
            )
        return result
