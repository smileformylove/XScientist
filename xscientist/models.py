from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


def _path_text(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(Path(value).expanduser())


@dataclass(frozen=True)
class ProjectRequest:
    """Stable Python API request for an end-to-end project run."""

    project: str
    question: str | None = None
    topic: str | Path | None = None
    ideas: str | Path | None = None
    output_root: str | Path | None = None
    num_ideas: int = 3
    parallel: bool = False
    num_workers: int = 2
    workflow_mode: str = "adaptive"
    target_venue: str | None = None
    submission_mode: bool = False
    breakthrough_mode: bool = False
    high_quality_mode: bool = False
    bfts_config: str | Path | None = None
    autopilot: str | None = None
    resume: bool = False
    data_dir: str | Path | None = None
    allow_synthetic_data: bool = False
    max_project_tokens: int | None = None
    max_project_hours: float | None = None
    max_cost_usd: float | None = None
    # ``research_git`` remains a compatibility spelling for releases <=0.1.
    research_git: str = "local"
    research_vcs: str | None = None
    git_checkpoint_policy: str = "milestone"
    research_git_strict: bool = False
    extra_args: Sequence[str] = field(default_factory=tuple)

    def to_argv(self) -> list[str]:
        if not str(self.project or "").strip():
            raise ValueError("project is required")
        sources = sum(
            value is not None for value in (self.question, self.topic, self.ideas)
        )
        if sources == 0 and not self.resume:
            raise ValueError("one of question, topic or ideas is required")
        if sources > 1:
            raise ValueError("question, topic, and ideas are mutually exclusive")
        if self.num_ideas < 1:
            raise ValueError("num_ideas must be at least 1")
        if self.num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        research_vcs = self.research_vcs or self.research_git
        if research_vcs not in {"off", "local"}:
            raise ValueError("research_vcs must be off or local")
        if self.git_checkpoint_policy not in {"manual", "stage", "milestone"}:
            raise ValueError(
                "git_checkpoint_policy must be manual, stage, or milestone"
            )
        if self.autopilot not in {None, "balanced", "discovery", "publication"}:
            raise ValueError(
                "autopilot must be balanced, discovery, publication, or None"
            )
        if self.data_dir is not None and self.allow_synthetic_data:
            raise ValueError("data_dir and allow_synthetic_data are mutually exclusive")
        for label, value in (
            ("max_project_tokens", self.max_project_tokens),
            ("max_project_hours", self.max_project_hours),
            ("max_cost_usd", self.max_cost_usd),
        ):
            if value is not None and float(value) <= 0:
                raise ValueError(f"{label} must be greater than zero")

        argv = [str(self.project)]
        if self.output_root is not None:
            argv.extend(["--output-root", _path_text(self.output_root) or ""])
        if self.question is not None:
            argv.extend(["--question", str(self.question)])
        if self.topic is not None:
            argv.extend(["--topic", _path_text(self.topic) or ""])
        if self.ideas is not None:
            argv.extend(["--ideas", _path_text(self.ideas) or ""])
        argv.extend(["--num-ideas", str(self.num_ideas)])
        argv.extend(["--workflow-mode", self.workflow_mode])
        if self.parallel:
            argv.append("--parallel")
            argv.extend(["--num-workers", str(self.num_workers)])
        if self.target_venue:
            argv.extend(["--target-venue", self.target_venue])
        if self.submission_mode:
            argv.append("--submission-mode")
        if self.breakthrough_mode:
            argv.append("--breakthrough-mode")
        if self.high_quality_mode:
            argv.append("--high-quality-mode")
        if self.bfts_config is not None:
            argv.extend(["--bfts-config", _path_text(self.bfts_config) or ""])
        if self.autopilot is not None:
            argv.extend(["--autopilot", self.autopilot])
        if self.resume:
            argv.append("--resume")
        if self.data_dir is not None:
            argv.extend(["--data-dir", _path_text(self.data_dir) or ""])
        if self.allow_synthetic_data:
            argv.append("--allow-synthetic-data")
        for flag, value in (
            ("--max-project-tokens", self.max_project_tokens),
            ("--max-project-hours", self.max_project_hours),
            ("--max-cost-usd", self.max_cost_usd),
        ):
            if value is not None:
                argv.extend([flag, str(value)])
        if research_vcs != "off":
            vcs_flag = (
                "--research-vcs" if self.research_vcs is not None else "--research-git"
            )
            policy_flag = (
                "--checkpoint-policy"
                if self.research_vcs is not None
                else "--git-checkpoint-policy"
            )
            argv.extend([vcs_flag, research_vcs])
            argv.extend([policy_flag, self.git_checkpoint_policy])
        if self.research_git_strict:
            argv.append(
                "--research-vcs-strict"
                if self.research_vcs is not None
                else "--research-git-strict"
            )
        argv.extend(str(arg) for arg in self.extra_args)
        return argv

    def to_dict(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "question": self.question,
            "topic": _path_text(self.topic),
            "ideas": _path_text(self.ideas),
            "output_root": _path_text(self.output_root),
            "num_ideas": self.num_ideas,
            "parallel": self.parallel,
            "num_workers": self.num_workers,
            "workflow_mode": self.workflow_mode,
            "target_venue": self.target_venue,
            "submission_mode": self.submission_mode,
            "breakthrough_mode": self.breakthrough_mode,
            "high_quality_mode": self.high_quality_mode,
            "bfts_config": _path_text(self.bfts_config),
            "autopilot": self.autopilot,
            "resume": self.resume,
            "data_dir": _path_text(self.data_dir),
            "allow_synthetic_data": self.allow_synthetic_data,
            "max_project_tokens": self.max_project_tokens,
            "max_project_hours": self.max_project_hours,
            "max_cost_usd": self.max_cost_usd,
            "research_git": self.research_git,
            "research_vcs": self.research_vcs,
            "git_checkpoint_policy": self.git_checkpoint_policy,
            "research_git_strict": self.research_git_strict,
            "extra_args": list(self.extra_args),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectRequest":
        return cls(**dict(payload))


@dataclass(frozen=True)
class CommandResult:
    """Result returned by SDK operations and background jobs."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    started_at: str
    finished_at: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "ok": self.ok,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CommandResult":
        data = dict(payload)
        data.pop("ok", None)
        data["command"] = tuple(data.get("command") or ())
        return cls(**data)


@dataclass(frozen=True)
class ServiceSettings:
    """Runtime settings for the HTTP service."""

    work_dir: str | Path | None = None
    output_root: str | Path | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    max_workers: int = 2
    max_output_chars: int = 200_000
    api_key: str | None = None
    state_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_output_chars < 1:
            raise ValueError("max_output_chars must be at least 1")
