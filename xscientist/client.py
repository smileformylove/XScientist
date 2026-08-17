from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from ai_scientist.resources import resolve_bfts_config_path
from ai_scientist.utils.bounded_process import (
    ProcessCancelled,
    ProcessResourceLimitExceeded,
    run_process_bounded,
    workspace_limit_checker,
)

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

    def _research_manager(self):
        from ai_scientist.apps.manager import ResearchManager

        return ResearchManager(
            str(self.output_root) if self.output_root is not None else None
        )

    @staticmethod
    def _validate_limit(value: int, *, label: str) -> int:
        limit = int(value)
        if limit < 1:
            raise ValueError(f"{label} must be at least 1")
        if limit > 1000:
            raise ValueError(f"{label} must not exceed 1000")
        return limit

    @staticmethod
    def _validate_paper_folder(folder: str, *, papers_root: Path) -> str:
        raw_name = str(folder or "")
        name = raw_name.strip()
        if (
            not name
            or name != raw_name
            or name in {".", ".."}
            or name.startswith(("-", "~"))
            or Path(name).is_absolute()
            or "/" in name
            or "\\" in name
            or any(ord(char) < 32 for char in raw_name)
        ):
            raise ValueError("folder must be a single paper directory name")
        candidate = (papers_root / name).resolve()
        try:
            candidate.relative_to(papers_root)
        except ValueError as exc:
            raise ValueError(
                "folder must stay within the configured output_root"
            ) from exc
        return name

    def list_papers(
        self,
        *,
        paper_type: str | None = None,
        sort_by: str = "modified",
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """List generated papers from the configured output root."""

        if sort_by not in {"modified", "quality"}:
            raise ValueError("sort_by must be 'modified' or 'quality'")
        validated_limit = self._validate_limit(limit, label="limit")
        items = self._research_manager().list_papers(
            paper_type=paper_type,
            sort_by=sort_by,
        )
        return items[:validated_limit]

    def get_paper(self, folder: str) -> dict[str, object] | None:
        """Load one generated paper by its folder under the configured output root."""

        manager = self._research_manager()
        papers_root = manager.papers_dir.resolve()
        name = self._validate_paper_folder(folder, papers_root=papers_root)
        return manager.get_paper_details(name)

    def shortlist_papers(
        self,
        *,
        paper_type: str | None = None,
        target_venue: str | None = None,
        require_gate: bool = False,
        require_ready: bool = False,
        min_breakthrough: float | None = None,
        min_priority: float | None = None,
        max_blockers: int | None = None,
        min_rewrite_gain: float | None = None,
        top_n: int = 5,
    ) -> list[dict[str, object]]:
        """Return the strongest submission candidates without mutating outputs."""

        validated_top_n = self._validate_limit(top_n, label="top_n")
        return self._research_manager().shortlist_papers(
            paper_type=paper_type,
            target_venue=target_venue,
            require_gate=require_gate,
            require_ready=require_ready,
            min_breakthrough=min_breakthrough,
            min_priority=min_priority,
            max_blockers=max_blockers,
            min_rewrite_gain=min_rewrite_gain,
            top_n=validated_top_n,
        )

    def submission_board(
        self,
        *,
        top_n_per_venue: int = 3,
        require_gate: bool = False,
        min_breakthrough: float | None = None,
        min_priority: float | None = None,
        max_blockers: int | None = None,
        min_rewrite_gain: float | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """Group submission candidates by target venue."""

        validated_top_n = self._validate_limit(top_n_per_venue, label="top_n_per_venue")
        return self._research_manager().submission_board(
            top_n_per_venue=validated_top_n,
            require_gate=require_gate,
            min_breakthrough=min_breakthrough,
            min_priority=min_priority,
            max_blockers=max_blockers,
            min_rewrite_gain=min_rewrite_gain,
        )

    def rewrite_board(
        self,
        *,
        top_n: int = 10,
        paper_type: str | None = None,
        target_venue: str | None = None,
        min_priority: float | None = None,
        min_rewrite_gain: float | None = None,
        max_blockers: int | None = None,
        require_gate: bool = False,
        include_ready: bool = False,
    ) -> list[dict[str, object]]:
        """Return papers ranked by the value of another focused rewrite."""

        validated_top_n = self._validate_limit(top_n, label="top_n")
        return self._research_manager().rewrite_board(
            top_n=validated_top_n,
            paper_type=paper_type,
            target_venue=target_venue,
            min_priority=min_priority,
            min_rewrite_gain=min_rewrite_gain,
            max_blockers=max_blockers,
            require_gate=require_gate,
            include_ready=include_ready,
        )

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
        max_output_chars: int | None = None,
        max_workspace_bytes: int | None = None,
        max_workspace_files: int | None = None,
        cancel_check=None,
        output_callback=None,
    ) -> CommandResult:
        output_root = request.output_root or self.output_root
        workspace = (
            Path(output_root).expanduser().resolve() / "projects" / request.project
            if output_root is not None
            else None
        )
        return self.run_command(
            self.project_command(request),
            check=check,
            timeout=timeout,
            max_output_chars=max_output_chars,
            workspace=workspace,
            max_workspace_bytes=max_workspace_bytes,
            max_workspace_files=max_workspace_files,
            cancel_check=cancel_check,
            output_callback=output_callback,
        )

    def run_command(
        self,
        command: Sequence[str],
        *,
        check: bool = False,
        timeout: float | None = None,
        max_output_chars: int | None = None,
        workspace: str | Path | None = None,
        max_workspace_bytes: int | None = None,
        max_workspace_files: int | None = None,
        cancel_check=None,
        output_callback=None,
    ) -> CommandResult:
        env = os.environ.copy()
        env.update(self.env)
        if self.output_root is not None:
            env.setdefault("RESEARCH_OUTPUT_DIR", str(self.output_root))
        started_at = _now_iso()
        limit_check = (
            workspace_limit_checker(
                workspace,
                max_bytes=max_workspace_bytes,
                max_files=max_workspace_files,
            )
            if workspace is not None
            and (max_workspace_bytes is not None or max_workspace_files is not None)
            else None
        )
        try:
            if (
                max_output_chars is None
                and cancel_check is None
                and output_callback is None
            ):
                completed = subprocess.run(
                    [str(item) for item in command],
                    cwd=str(self.work_dir) if self.work_dir is not None else None,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                stdout_truncated = False
                stderr_truncated = False
            else:
                completed = run_process_bounded(
                    command,
                    cwd=self.work_dir,
                    env=env,
                    timeout=timeout,
                    max_output_chars=max_output_chars or 200_000,
                    limit_check=limit_check,
                    cancel_check=cancel_check,
                    output_callback=output_callback,
                )
                returncode = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                stdout_truncated = completed.stdout_truncated
                stderr_truncated = completed.stderr_truncated
        except ProcessCancelled as exc:
            returncode = 130
            stdout = exc.stdout
            stderr = (exc.stderr + "\n" if exc.stderr else "") + "RunCancelled"
            stdout_truncated = exc.stdout_truncated
            output_limit = int(max_output_chars or 200_000)
            stderr_truncated = exc.stderr_truncated or len(stderr) > output_limit
            stderr = stderr[-output_limit:]
        except ProcessResourceLimitExceeded as exc:
            returncode = 75
            stdout = exc.stdout
            stderr = (exc.stderr + "\n" if exc.stderr else "") + (
                f"ResourceLimitError: {exc.reason}"
            )
            stdout_truncated = len(stdout) >= int(max_output_chars or 1)
            stderr_truncated = len(stderr) > int(max_output_chars or len(stderr))
            if max_output_chars is not None:
                stderr = stderr[-max_output_chars:]
        result = CommandResult(
            command=tuple(str(item) for item in command),
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            started_at=started_at,
            finished_at=_now_iso(),
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )
        if check and not result.ok:
            raise subprocess.CalledProcessError(
                result.returncode,
                list(result.command),
                output=result.stdout,
                stderr=result.stderr,
            )
        return result
