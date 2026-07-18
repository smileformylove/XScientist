"""configuration and setup utils"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Hashable, cast, Literal, Optional

import coolname
import rich
from omegaconf import OmegaConf
from rich.syntax import Syntax
import shutup
from rich.logging import RichHandler
import logging

from . import tree_export
from . import copytree, preproc_data, serialize

shutup.mute_warnings()
logging.basicConfig(
    level="WARNING", format="%(message)s", datefmt="[%X]", handlers=[RichHandler()]
)
logger = logging.getLogger("ai-scientist")
logger.setLevel(logging.WARNING)


""" these dataclasses are just for type hinting, the actual config is in config.yaml """


@dataclass
class ThinkingConfig:
    type: str
    budget_tokens: Optional[int] = None


@dataclass
class StageConfig:
    model: str
    temp: float
    thinking: ThinkingConfig
    betas: str
    max_tokens: Optional[int] = None


@dataclass
class SearchConfig:
    max_debug_depth: int
    debug_prob: float
    num_drafts: int


@dataclass
class DebugConfig:
    stage4: bool


@dataclass
class AgentConfig:
    steps: int
    stages: dict[str, int]
    k_fold_validation: int
    expose_prediction: bool
    data_preview: bool

    code: StageConfig
    feedback: StageConfig
    vlm_feedback: StageConfig

    search: SearchConfig
    num_workers: int
    type: str
    multi_seed_eval: dict[str, int]

    summary: Optional[StageConfig] = None
    select_node: Optional[StageConfig] = None


@dataclass
class ExecConfig:
    timeout: int
    agent_file_name: str
    format_tb_ipython: bool
    backend: str = "auto"
    require_isolation: bool = False
    docker_image: str = "xscientist-exec:latest"
    network: str = "none"
    allow_experiment_network: bool = False
    memory: str = "4g"
    cpus: float = 2.0
    pids_limit: int = 256
    read_only_root: bool = True
    read_only_mounts: list[str] = field(default_factory=list)


@dataclass
class ExperimentConfig:
    num_syn_datasets: int


@dataclass
class LLMBudgetConfig:
    max_total_tokens: Optional[int] = None
    max_cost_usd: Optional[float] = None
    max_wall_time_seconds: Optional[float] = None
    prices_per_million: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class Config(Hashable):
    data_dir: Path
    desc_file: Path | None

    goal: str | None
    eval: str | None

    log_dir: Path
    workspace_dir: Path

    preprocess_data: bool
    copy_data: bool

    exp_name: str

    exec: ExecConfig
    generate_report: bool
    report: StageConfig
    agent: AgentConfig
    experiment: ExperimentConfig
    debug: DebugConfig
    llm_budget: LLMBudgetConfig = field(default_factory=LLMBudgetConfig)
    resume_from: Optional[Path] = None


def _get_next_logindex(dir: Path) -> int:
    """Get the next available index for a log directory."""
    max_index = -1
    for p in dir.iterdir():
        try:
            if (current_index := int(p.name.split("-")[0])) > max_index:
                max_index = current_index
        except ValueError:
            pass
    print("max_index: ", max_index)
    return max_index + 1


def _load_cfg(
    path: Path = Path(__file__).parent / "config.yaml", use_cli_args=False
) -> Config:
    cfg = OmegaConf.load(path)
    if use_cli_args:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())
    return cfg


def load_cfg(path: Path = Path(__file__).parent / "config.yaml") -> Config:
    """Load config from .yaml file and CLI args, and set up logging directory."""
    path = Path(path).expanduser().resolve()
    return prep_cfg(_load_cfg(path), base_dir=path.parent)


def _resolve_config_path(value, *, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def prep_cfg(cfg: Config, *, base_dir: str | Path | None = None):
    base_dir = Path(base_dir or Path.cwd()).expanduser().resolve()
    if cfg.data_dir is None:
        raise ValueError("`data_dir` must be provided.")

    if cfg.desc_file is None and cfg.goal is None:
        raise ValueError(
            "You must provide either a description of the task goal (`goal=...`) or a path to a plaintext file containing the description (`desc_file=...`)."
        )

    cfg.data_dir = _resolve_config_path(cfg.data_dir, base_dir=base_dir)

    if cfg.desc_file is not None:
        cfg.desc_file = _resolve_config_path(cfg.desc_file, base_dir=base_dir)

    top_log_dir = _resolve_config_path(cfg.log_dir, base_dir=base_dir)
    top_log_dir.mkdir(parents=True, exist_ok=True)

    top_workspace_dir = _resolve_config_path(cfg.workspace_dir, base_dir=base_dir)
    top_workspace_dir.mkdir(parents=True, exist_ok=True)

    resume_from = getattr(cfg, "resume_from", None)
    if resume_from:
        resume_from = _resolve_config_path(resume_from, base_dir=base_dir)
        if not resume_from.is_file():
            raise FileNotFoundError(f"BFTS checkpoint not found: {resume_from}")
        run_name = resume_from.parent.parent.name
        cfg.exp_name = run_name
        cfg.log_dir = (top_log_dir / run_name).resolve()
        cfg.workspace_dir = (top_workspace_dir / run_name).resolve()
        if not cfg.workspace_dir.is_dir():
            raise FileNotFoundError(
                f"BFTS workspace for checkpoint does not exist: {cfg.workspace_dir}"
            )
        cfg.resume_from = resume_from
    else:
        # generate experiment name and prefix with consecutive index
        ind = max(
            _get_next_logindex(top_log_dir), _get_next_logindex(top_workspace_dir)
        )
        cfg.exp_name = cfg.exp_name or coolname.generate_slug(3)
        cfg.exp_name = f"{ind}-{cfg.exp_name}"

        cfg.log_dir = (top_log_dir / cfg.exp_name).resolve()
        cfg.workspace_dir = (top_workspace_dir / cfg.exp_name).resolve()

    from ai_scientist.utils.llm_budget import configure_llm_budget

    budget_cfg = getattr(cfg, "llm_budget", None)
    config_prices = dict(getattr(budget_cfg, "prices_per_million", {}) or {})
    env_prices = {}
    if not config_prices:
        try:
            env_prices = json.loads(
                os.environ.get("AI_SCIENTIST_LLM_PRICES_JSON", "{}")
            )
        except json.JSONDecodeError:
            env_prices = {}
    configured_state_path = os.environ.get("AI_SCIENTIST_LLM_BUDGET_STATE")
    configure_llm_budget(
        max_total_tokens=(
            getattr(budget_cfg, "max_total_tokens", None)
            if getattr(budget_cfg, "max_total_tokens", None) is not None
            else os.environ.get("AI_SCIENTIST_LLM_MAX_TOTAL_TOKENS")
        ),
        max_cost_usd=(
            getattr(budget_cfg, "max_cost_usd", None)
            if getattr(budget_cfg, "max_cost_usd", None) is not None
            else os.environ.get("AI_SCIENTIST_LLM_MAX_COST_USD")
        ),
        max_wall_time_seconds=(
            getattr(budget_cfg, "max_wall_time_seconds", None)
            if getattr(budget_cfg, "max_wall_time_seconds", None) is not None
            else os.environ.get("AI_SCIENTIST_LLM_MAX_WALL_TIME_SECONDS")
        ),
        prices_per_million=config_prices or env_prices,
        state_path=(
            cfg.workspace_dir / "llm_budget.json"
            if resume_from
            else (
                configured_state_path
                if configured_state_path
                else cfg.workspace_dir / "llm_budget.json"
            )
        ),
        reset=not bool(configured_state_path) and not bool(resume_from),
        allow_limit_increase=bool(resume_from),
        reclaim_active_reservations=bool(resume_from),
    )

    # validate the config
    cfg_schema: Config = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(cfg_schema, cfg)

    if cfg.agent.type not in ["parallel", "sequential"]:
        raise ValueError("agent.type must be either 'parallel' or 'sequential'")

    return cast(Config, cfg)


def print_cfg(cfg: Config) -> None:
    rich.print(Syntax(OmegaConf.to_yaml(cfg), "yaml", theme="paraiso-dark"))


def load_task_desc(cfg: Config):
    """Load task description from markdown file or config str."""

    # either load the task description from a file
    if cfg.desc_file is not None:
        if not (cfg.goal is None and cfg.eval is None):
            logger.warning(
                "Ignoring goal and eval args because task description file is provided."
            )

        with open(cfg.desc_file) as f:
            return f.read()

    # or generate it from the goal and eval args
    if cfg.goal is None:
        raise ValueError(
            "`goal` (and optionally `eval`) must be provided if a task description file is not provided."
        )

    task_desc = {"Task goal": cfg.goal}
    if cfg.eval is not None:
        task_desc["Task evaluation"] = cfg.eval
    print(task_desc)
    return task_desc


def prep_agent_workspace(cfg: Config):
    """Setup the agent's workspace and preprocess data if necessary."""
    (cfg.workspace_dir / "input").mkdir(parents=True, exist_ok=True)
    (cfg.workspace_dir / "working").mkdir(parents=True, exist_ok=True)

    copytree(cfg.data_dir, cfg.workspace_dir / "input", use_symlinks=not cfg.copy_data)
    if cfg.preprocess_data:
        preproc_data(cfg.workspace_dir / "input")


def save_run(
    cfg: Config,
    journal,
    stage_name: str = None,
    *,
    allow_llm_selection: bool = True,
):
    if stage_name is None:
        stage_name = "NoStageRun"
    save_dir = cfg.log_dir / stage_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # save journal
    try:
        serialize.dump_json(journal, save_dir / "journal.json")
    except Exception as e:
        print(f"Error saving journal: {e}")
        raise
    # save config
    try:
        OmegaConf.save(config=cfg, f=save_dir / "config.yaml")
    except Exception as e:
        print(f"Error saving config: {e}")
        raise
    # create the tree + code visualization
    try:
        tree_export.generate(cfg, journal, save_dir / "tree_plot.html")
    except Exception as e:
        print(f"Error generating tree: {e}")
        raise
    # save the best found solution
    try:
        # Prefer deterministic metric-based selection for performance and stability.
        best_node = journal.get_best_node_by_metric(only_good=True)
        if best_node is None and allow_llm_selection:
            best_node = journal.get_best_node(only_good=False, cfg=cfg)
        if best_node is not None:
            for existing_file in save_dir.glob("best_solution_*.py"):
                existing_file.unlink()
            # Create new best solution file
            filename = f"best_solution_{best_node.id}.py"
            with open(save_dir / filename, "w") as f:
                f.write(best_node.code)
            # save best_node.id to a text file
            with open(save_dir / "best_node_id.txt", "w") as f:
                f.write(str(best_node.id))
        else:
            print("No best node found yet")
    except Exception as e:
        print(f"Error saving best solution: {e}")
