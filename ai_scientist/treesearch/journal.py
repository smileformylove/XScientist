from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional, Any
import copy
import math
import os
import re

from dataclasses_json import DataClassJsonMixin
from .interpreter import ExecutionResult
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import trim_long_string
from .utils.serialize import atomic_write_json, atomic_write_text
from .backend import (
    FunctionCallValidationError,
    FunctionSpec,
    ResearchDecisionError,
    query,
)

from rich import print

import logging
from pathlib import Path
from ai_scientist.utils.llm_budget import is_llm_budget_exception
from ai_scientist.utils.evaluation_binding import (
    evaluation_comparison_contract,
    evaluation_hash_binding,
)

logger = logging.getLogger(__name__)

NODE_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
NODE_ID_RE = re.compile(NODE_ID_PATTERN)


def _optional_config_value(config: Any, key: str) -> Any:
    """Read an optional field from mapping-, OmegaConf-, or object-style config."""

    if config is None:
        return None
    getter = getattr(config, "get", None)
    if callable(getter):
        try:
            return getter(key, None)
        except TypeError:
            # Some object-like configs expose ``get(key)`` without a default.
            return getter(key)
    return getattr(config, key, None)


def _experiment_notes_summary_route(cfg: Any) -> tuple[str, float, int | None]:
    """Resolve the stage-notes model without inventing a provider route.

    The explicit precedence is ``agent.summary``, then ``report``, then the
    operator-provided ``ZHIPU_DEFAULT_MODEL`` environment variable. A missing
    route fails before any note artifact is written instead of silently using a
    hard-coded model/provider combination.
    """

    agent_cfg = _optional_config_value(cfg, "agent")
    summary_cfg = _optional_config_value(agent_cfg, "summary")
    route_name = "cfg.agent.summary"
    if summary_cfg is None:
        summary_cfg = _optional_config_value(cfg, "report")
        route_name = "cfg.report"

    if summary_cfg is not None:
        model = _optional_config_value(summary_cfg, "model")
        temperature = _optional_config_value(summary_cfg, "temp")
        max_tokens = _optional_config_value(summary_cfg, "max_tokens")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"{route_name}.model must be a non-empty string")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
        ):
            raise ValueError(f"{route_name}.temp must be a finite number")
        if max_tokens is not None and (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ValueError(
                f"{route_name}.max_tokens must be a positive integer or null"
            )
        return model.strip(), float(temperature), max_tokens

    environment_model = os.environ.get("ZHIPU_DEFAULT_MODEL", "").strip()
    if environment_model:
        return environment_model, 0.3, None
    raise ValueError(
        "Experiment-note summary requires cfg.agent.summary, cfg.report, "
        "or ZHIPU_DEFAULT_MODEL"
    )


def _validate_node_id(node_id: Any, *, context: str = "Node") -> str:
    if not isinstance(node_id, str) or NODE_ID_RE.fullmatch(node_id) is None:
        raise ValueError(f"{context} has an invalid id")
    return node_id


def _safe_relative_to(path: Path) -> Path:
    try:
        return path.relative_to(os.getcwd())
    except ValueError:
        return path


node_selection_spec = FunctionSpec(
    name="select_best_implementation",
    description="Select the best implementation based on comprehensive analysis",
    json_schema={
        "type": "object",
        "properties": {
            "selected_id": {
                "type": "string",
                "description": "ID of the selected best implementation",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed explanation of why this implementation was chosen",
            },
        },
        "required": ["selected_id", "reasoning"],
        "additionalProperties": False,
    },
)


@dataclass(eq=False)
class Node(DataClassJsonMixin):
    """A single node in the solution tree. Contains code, execution results, and evaluation information."""

    # ---- code & plan ----
    plan: str = field(default="", kw_only=True)  # type: ignore
    overall_plan: str = field(default="", kw_only=True)  # type: ignore
    code: str = field(default="", kw_only=True)  # type: ignore
    plot_code: str = field(default=None, kw_only=True)  # type: ignore
    plot_plan: str = field(default=None, kw_only=True)  # type: ignore

    # ---- general attrs ----
    step: int = field(default=None, kw_only=True)  # type: ignore
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=lambda: time.time(), kw_only=True)
    parent: Optional["Node"] = field(default=None, kw_only=True)
    children: set["Node"] = field(default_factory=set, kw_only=True)
    exp_results_dir: str = field(default=None, kw_only=True)  # type: ignore

    # ---- execution info ----
    _term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    exec_time: float = field(default=None, kw_only=True)  # type: ignore
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)
    exc_stack: list[tuple] | None = field(default=None, kw_only=True)
    execution_backend: str | None = field(default=None, kw_only=True)
    execution_isolation: dict[str, Any] | None = field(default=None, kw_only=True)

    # ---- parsing info ----
    parse_metrics_plan: str = field(default="", kw_only=True)
    parse_metrics_code: str = field(default="", kw_only=True)
    # parse_exec_result: ExecutionResult = field(default=None, kw_only=True)
    parse_term_out: list[str] = field(default=None, kw_only=True)
    parse_exc_type: str | None = field(default=None, kw_only=True)
    parse_exc_info: dict | None = field(default=None, kw_only=True)
    parse_exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- plot execution info ----
    plot_term_out: list[str] = field(default=None, kw_only=True)  # type: ignore
    plot_exec_time: float = field(default=None, kw_only=True)  # type: ignore
    plot_exc_type: str | None = field(default=None, kw_only=True)
    plot_exc_info: dict | None = field(default=None, kw_only=True)
    plot_exc_stack: list[tuple] | None = field(default=None, kw_only=True)

    # ---- evaluation ----
    # post-execution result analysis (findings/feedback)
    analysis: str = field(default=None, kw_only=True)  # type: ignore
    agent_review_bug_advisory: bool | None = field(default=None, kw_only=True)
    metric: MetricValue = field(default=None, kw_only=True)  # type: ignore
    metric_provenance: str = field(default="unavailable", kw_only=True)
    advisory_metric: dict[str, Any] | None = field(default=None, kw_only=True)
    evaluation_report: dict[str, Any] | None = field(default=None, kw_only=True)
    multi_seed_report: dict[str, Any] | None = field(default=None, kw_only=True)
    multi_seed_attempts: list[dict[str, Any]] = field(
        default_factory=list, kw_only=True
    )
    # whether the agent decided that the code is buggy
    # -> always True if exc_type is not None or no valid metric
    is_buggy: bool = field(default=None, kw_only=True)  # type: ignore
    is_buggy_plots: bool = field(default=None, kw_only=True)

    # ---- plotting ----
    plot_data: dict = field(default_factory=dict, kw_only=True)
    plots_generated: bool = field(default=False, kw_only=True)
    plots: List[str] = field(default_factory=list)  # Relative paths for visualization
    plot_paths: List[str] = field(
        default_factory=list
    )  # Absolute paths for programmatic access

    # ---- VLM feedback ----
    plot_analyses: List[str] = field(default_factory=list)
    vlm_feedback_summary: List[str] = field(default_factory=list)
    datasets_successfully_tested: List[str] = field(default_factory=list)

    # ---- execution time feedback ----
    exec_time_feedback: str = field(default="", kw_only=True)

    # ---- ablation study ----
    ablation_name: str = field(default=None, kw_only=True)
    ablation_control_node_id: str | None = field(default=None, kw_only=True)
    ablation_component: str | None = field(default=None, kw_only=True)
    ablation_expected_outcome: str | None = field(default=None, kw_only=True)
    ablation_code_diff_hash: str | None = field(default=None, kw_only=True)
    ablation_control_semantic_hash: str | None = field(default=None, kw_only=True)
    ablation_semantic_hash: str | None = field(default=None, kw_only=True)

    # ---- hyperparam tuning ----
    hyperparam_name: str = field(default=None, kw_only=True)

    # ---- seed node ----
    is_seed_node: bool = field(default=False, kw_only=True)
    is_seed_agg_node: bool = field(default=False, kw_only=True)
    random_seed: int | None = field(default=None, kw_only=True)
    seed_bootstrap_hash: str | None = field(default=None, kw_only=True)

    # ---- LLM provenance ----
    # Semantic call_receipt_ref.hash values from <ara>/llm/calls.jsonl for calls that
    # produced this node's code/plan. Left empty on nodes built without an
    # active tracer (legacy runs, seed nodes, deterministic replays).
    # Consumed by export_ara → hash_node_payload(llm_call_hashes=...) so
    # two nodes with identical code but different prompts hash differently.
    llm_call_refs: list[str] = field(default_factory=list, kw_only=True)

    # ContextPack object hashes injected immediately before this node was
    # planned. This answers "what history did the agent actually see?" without
    # copying the pack into every node or changing the experiment content hash.
    context_pack_refs: list[str] = field(default_factory=list, kw_only=True)

    def __post_init__(self) -> None:
        _validate_node_id(self.id)
        # Ensure children is a set even if initialized with a list
        if isinstance(self.children, list):
            self.children = set(self.children)
        # Only try to add to parent's children if parent is a Node object
        if self.parent is not None and not isinstance(self.parent, str):
            self.parent.children.add(self)

    @property
    def has_verified_metric(self) -> bool:
        """Return whether metric provenance is bound to a valid evaluator receipt."""

        if (
            self.metric_provenance != "deterministic_verified"
            or not isinstance(self.metric, MetricValue)
            or not self.metric.is_valid
            or not isinstance(self.evaluation_report, dict)
            or evaluation_hash_binding(self.evaluation_report) is None
        ):
            return False
        return self.evaluation_report.get("metric") == self.metric.value

    @property
    def evaluation_comparison_contract(self) -> dict[str, Any] | None:
        if not self.has_verified_metric:
            return None
        return evaluation_comparison_contract(self.evaluation_report)

    def __deepcopy__(self, memo):
        # Create a new instance with copied attributes
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        # Copy all attributes except parent and children to avoid circular references
        for k, v in self.__dict__.items():
            if k not in ("parent", "children"):
                setattr(result, k, copy.deepcopy(v, memo))

        # Handle parent and children separately
        result.parent = self.parent  # Keep the same parent reference
        result.children = set()  # Start with empty children set

        return result

    def __getstate__(self):
        """Return state for pickling"""
        state = self.__dict__.copy()
        # Ensure id is included in the state
        if hasattr(self, "id"):
            state["id"] = self.id
        return state

    def __setstate__(self, state):
        """Set state during unpickling"""
        # Ensure all required attributes are present
        self.__dict__.update(state)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """
        Return the stage of the node:
        - "stage" if the node is an initial solution draft
        - "debug" if the node is the result of a debugging step
        - "improve" if the node is the result of an improvement step
        """
        if self.parent is None:
            return "draft"
        return "debug" if self.parent.is_buggy else "improve"

    def absorb_exec_result(self, exec_result: ExecutionResult):
        """Absorb the result of executing the code from this node."""
        self._term_out = exec_result.term_out
        self.exec_time = exec_result.exec_time
        self.exc_type = exec_result.exc_type
        self.exc_info = exec_result.exc_info
        self.exc_stack = exec_result.exc_stack
        self.execution_backend = exec_result.execution_backend
        self.execution_isolation = exec_result.isolation

    def absorb_plot_exec_result(self, plot_exec_result: ExecutionResult):
        """Absorb the result of executing the plotting code from this node."""
        self.plot_term_out = plot_exec_result.term_out
        self.plot_exec_time = plot_exec_result.exec_time
        self.plot_exc_type = plot_exec_result.exc_type
        self.plot_exc_info = plot_exec_result.exc_info
        self.plot_exc_stack = plot_exec_result.exc_stack

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution (after truncating it)."""
        return trim_long_string("".join(self._term_out))

    @property
    def is_leaf(self) -> bool:
        """Check if the node is a leaf node in the solution tree."""
        return not self.children

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def debug_depth(self) -> int:
        """
        Length of the current debug path
        - 0 if the node is not a debug node (parent is not buggy)
        - 1 if the parent is buggy but the skip parent isn't
        - n if there were n consecutive debugging steps
        """
        if self.stage_name != "debug":
            return 0
        return self.parent.debug_depth + 1  # type: ignore

    def to_dict(self, *, artifact_base: Path | None = None) -> Dict:
        """Convert node to dictionary for serialization"""
        serialized_results_dir = None
        if self.exp_results_dir:
            results_path = Path(self.exp_results_dir)
            if not results_path.is_absolute():
                results_path = Path.cwd() / results_path
            results_path = results_path.resolve()
            if artifact_base is not None:
                try:
                    results_path = results_path.relative_to(artifact_base.resolve())
                except ValueError:
                    pass
            else:
                results_path = _safe_relative_to(results_path)
            serialized_results_dir = str(results_path)
        return {
            "code": self.code,
            "plan": self.plan,
            "overall_plan": (
                self.overall_plan if hasattr(self, "overall_plan") else None
            ),
            "plot_code": self.plot_code,
            "plot_plan": self.plot_plan,
            "step": self.step,
            "id": self.id,
            "ctime": self.ctime,
            "_term_out": self._term_out,
            "parse_metrics_plan": self.parse_metrics_plan,
            "parse_metrics_code": self.parse_metrics_code,
            "parse_term_out": self.parse_term_out,
            "parse_exc_type": self.parse_exc_type,
            "parse_exc_info": self.parse_exc_info,
            "parse_exc_stack": self.parse_exc_stack,
            "exec_time": self.exec_time,
            "exc_type": self.exc_type,
            "exc_info": self.exc_info,
            "exc_stack": self.exc_stack,
            "execution_backend": self.execution_backend,
            "execution_isolation": self.execution_isolation,
            "analysis": self.analysis,
            "agent_review_bug_advisory": self.agent_review_bug_advisory,
            "exp_results_dir": serialized_results_dir,
            "metric": {
                "value": self.metric.value if self.metric else None,
                "maximize": self.metric.maximize if self.metric else None,
                "name": self.metric.name if hasattr(self.metric, "name") else None,
                "description": (
                    self.metric.description
                    if hasattr(self.metric, "description")
                    else None
                ),
            },
            "metric_provenance": self.metric_provenance,
            "advisory_metric": self.advisory_metric,
            "evaluation_report": self.evaluation_report,
            "multi_seed_report": self.multi_seed_report,
            "multi_seed_attempts": self.multi_seed_attempts,
            "is_buggy": self.is_buggy,
            "is_buggy_plots": self.is_buggy_plots,
            "parent_id": None if self.parent is None else self.parent.id,
            "children": [child.id for child in self.children] if self.children else [],
            "plot_data": self.plot_data,
            "plots_generated": self.plots_generated,
            "plots": self.plots,
            "plot_paths": (
                [str(_safe_relative_to(Path(p).resolve())) for p in self.plot_paths]
                if self.plot_paths
                else []
            ),
            "plot_analyses": [
                {
                    **analysis,
                    "plot_path": (
                        str(_safe_relative_to(Path(analysis["plot_path"]).resolve()))
                        if analysis.get("plot_path")
                        else None
                    ),
                }
                for analysis in self.plot_analyses
            ],
            "vlm_feedback_summary": self.vlm_feedback_summary,
            "datasets_successfully_tested": self.datasets_successfully_tested,
            "ablation_name": self.ablation_name,
            "ablation_control_node_id": self.ablation_control_node_id,
            "ablation_component": self.ablation_component,
            "ablation_expected_outcome": self.ablation_expected_outcome,
            "ablation_code_diff_hash": self.ablation_code_diff_hash,
            "ablation_control_semantic_hash": self.ablation_control_semantic_hash,
            "ablation_semantic_hash": self.ablation_semantic_hash,
            "hyperparam_name": self.hyperparam_name,
            "is_seed_node": self.is_seed_node,
            "is_seed_agg_node": self.is_seed_agg_node,
            "random_seed": self.random_seed,
            "seed_bootstrap_hash": self.seed_bootstrap_hash,
            "exec_time_feedback": self.exec_time_feedback,
            "llm_call_refs": list(self.llm_call_refs or []),
            "context_pack_refs": list(self.context_pack_refs or []),
            "plot_term_out": self.plot_term_out,
            "plot_exec_time": self.plot_exec_time,
            "plot_exc_type": self.plot_exc_type,
            "plot_exc_info": self.plot_exc_info,
            "plot_exc_stack": self.plot_exc_stack,
        }

    @classmethod
    def from_dict(cls, data: Dict, journal: Optional[Journal] = None) -> "Node":
        """Create a Node from a dictionary, optionally linking to journal for relationships"""
        # Remove relationship IDs from constructor data
        parent_id = data.pop("parent_id", None)
        children = data.pop("children", [])

        # Handle metric conversion
        metric_data = data.pop("metric", None)
        if metric_data:
            if isinstance(metric_data, dict):
                data["metric"] = MetricValue(
                    value=metric_data["value"],
                    maximize=metric_data["maximize"],
                    name=metric_data["name"],
                    description=metric_data["description"],
                )
            else:
                # Handle legacy format or None
                data["metric"] = (
                    WorstMetricValue()
                    if data.get("is_buggy")
                    else MetricValue(metric_data)
                )

        # Create node instance
        node = cls(**data)

        # If journal is provided, restore relationships
        if journal is not None and parent_id:
            parent = journal.get_node_by_id(parent_id)
            if parent:
                node.parent = parent
                parent.children.add(node)

        return node


@dataclass
class InteractiveSession(DataClassJsonMixin):
    """
    A collection of nodes for an interaction session
    (when the agent interacts with a Jupyter notebook-like interface).
    """

    nodes: list[Node] = field(default_factory=list)
    completed: bool = False

    def append(self, node: Node) -> None:
        node.step = len(self.nodes)
        self.nodes.append(node)

    def generate_nb_trace(self, include_prompt, comment_headers=True) -> str:
        """Generate a trace of the interactive session in IPython format."""
        trace = []
        header_prefix = "## " if comment_headers else ""
        for n in self.nodes:
            trace.append(f"\n{header_prefix}In [{n.step+1}]:\n")
            trace.append(n.code)
            trace.append(f"\n{header_prefix}Out [{n.step+1}]:\n")
            trace.append(n.term_out)

        if include_prompt and self.nodes:
            trace.append(f"\n{header_prefix}In [{self.nodes[-1].step+2}]:\n")

        return "\n".join(trace).strip()


@dataclass
class Journal:
    """A collection of nodes representing the solution tree."""

    nodes: list[Node] = field(default_factory=list)

    def __getitem__(self, idx: int) -> Node:
        return self.nodes[idx]

    def __len__(self) -> int:
        """Return the number of nodes in the journal."""
        return len(self.nodes)

    def append(self, node: Node) -> None:
        """Append a new node to the journal."""
        _validate_node_id(node.id)
        if any(existing.id == node.id for existing in self.nodes):
            raise ValueError(f"Journal contains duplicate node id: {node.id}")
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> list[Node]:
        """Return a list of nodes representing intial coding drafts"""
        return [n for n in self.nodes if n.parent is None]

    @property
    def buggy_nodes(self) -> list[Node]:
        """Return a list of nodes that are considered buggy by the agent."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> list[Node]:
        """Return a list of nodes that are not considered buggy by the agent."""
        list_of_nodes = [
            [
                n.step,
                n.parent.step if n.parent else None,
                n.id,
                n.is_buggy,
                n.is_buggy_plots,
            ]
            for n in self.nodes
        ]
        print(
            f"[purple]all nodes ID and is_buggy/is_buggy_plots flags: {list_of_nodes}[/purple]"
        )
        return [n for n in self.nodes if n.is_buggy is False]

    @property
    def verified_nodes(self) -> list[Node]:
        """Return runnable nodes backed by a finite deterministic metric."""

        return [node for node in self.good_nodes if node.has_verified_metric]

    def get_node_by_id(self, node_id: str) -> Optional[Node]:
        """Get a node by its ID."""
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_metric_history(self) -> list[MetricValue]:
        """Return a list of all metric values in the journal."""
        return [n.metric for n in self.nodes]

    def get_best_node(
        self, only_good=True, use_val_metric_only=False, cfg=None
    ) -> None | Node:
        """Return the best solution found so far."""
        if only_good:
            nodes = self.verified_nodes
            if not nodes:
                return None
        else:
            nodes = [node for node in self.nodes if node.has_verified_metric]

        if use_val_metric_only:
            return max(nodes, key=lambda n: n.metric)

        if len(nodes) == 1:
            return nodes[0]

        # Create evaluation prompt for LLM
        prompt = {
            "Introduction": (
                "You are an experienced AI researcher evaluating different implementations "
                "of an experiment to select the best one. You should consider all aspects "
                "including performance metrics, training dynamics, generated plots quality."
            ),
            "Task": (
                "Select the best implementation from the candidates below, considering all available evidence."
                "Avoid relying too heavily on the validation loss alone, because "
                "it may not be directly comparable across different objective functions or training details. "
                "If there are multiple validation losses (e.g., when evaluating multiple datasets), "
                "consider all of them and select the implementation that performs best overall."
            ),
            "Candidates": "",
        }
        # Gather info about each node
        for node in nodes:
            if not node.is_seed_node:
                candidate_info = (
                    f"ID: {node.id}\n" f"Metric: {str(node.metric)}\n"
                    if node.metric
                    else (
                        "N/A\n" f"Training Analysis: {node.analysis}\n"
                        if hasattr(node, "analysis")
                        else (
                            "N/A\n" f"VLM Feedback: {node.vlm_feedback_summary}\n"
                            if hasattr(node, "vlm_feedback_summary")
                            else "N/A\n"
                        )
                    )
                )
                prompt["Candidates"] += candidate_info

        try:
            if cfg is None or cfg.agent.get("select_node", None) is None:
                return self.get_best_node_by_metric()
            else:
                model = cfg.agent.select_node.model
                temperature = cfg.agent.select_node.temp
            selection = query(
                system_message=prompt,
                user_message=None,
                func_spec=node_selection_spec,
                model=model,
                temperature=temperature,
                max_tokens=_optional_config_value(cfg.agent.select_node, "max_tokens"),
            )

            # Find and return the selected node
            selected_node = next(
                (node for node in nodes if str(node.id) == selection["selected_id"]),
                None,
            )
            if selected_node:
                logger.warning(
                    f"Selected node {selected_node.id} as best implementation"
                )
                logger.warning("Research-agent node selection validated")
                return selected_node
            else:
                raise FunctionCallValidationError(
                    "Selected implementation is not an allowed candidate"
                )

        except Exception as e:
            if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                raise
            logger.error("Research-agent node selection failed: %s", type(e).__name__)
            raise ResearchDecisionError(
                "Research-agent node selection failed"
            ) from None

    def get_best_node_by_metric(
        self,
        *,
        only_good: bool = True,
        include_seed_nodes: bool = False,
        include_seed_agg_nodes: bool = False,
        reference_contract: dict[str, Any] | None = None,
    ) -> Optional[Node]:
        """
        Deterministically pick the best node by MetricValue comparison.

        This avoids extra LLM calls and is useful for progress tracking / gating.
        By default we ignore seed evaluation nodes so the "best" node corresponds to a
        first-class candidate implementation rather than a resampled rerun.
        """
        nodes = self.verified_nodes if only_good else self.nodes
        nodes = [node for node in nodes if node.has_verified_metric]
        if not include_seed_nodes:
            nodes = [n for n in nodes if not getattr(n, "is_seed_node", False)]
        if not include_seed_agg_nodes:
            nodes = [n for n in nodes if not getattr(n, "is_seed_agg_node", False)]
        if reference_contract is not None:
            nodes = [
                node
                for node in nodes
                if node.evaluation_comparison_contract == reference_contract
            ]
        if not nodes:
            return None
        # Metric family is locked before considering evidence coverage. A
        # wider but scientifically different metric must never win by breadth.
        families = {node.metric.comparison_family for node in nodes}
        if len(families) != 1:
            raise ResearchDecisionError(
                "Verified metrics use incompatible comparison contracts"
            )
        signatures = {node.metric.comparison_signature for node in nodes}
        if len(signatures) != 1:
            raise ResearchDecisionError(
                "Verified metrics use incompatible comparison contracts"
            )
        if len(nodes) > 1:
            locked_contract = nodes[0].evaluation_comparison_contract
            if locked_contract is None or any(
                node.evaluation_comparison_contract != locked_contract
                for node in nodes[1:]
            ):
                raise ResearchDecisionError(
                    "Verified evaluations use incompatible dataset identities"
                )
        return max(nodes, key=lambda node: node.metric)

    def generate_summary(self, include_code: bool = False, **model_kwargs) -> str:
        """Generate a summary of the research progress using LLM, including both successes and failures."""
        if not self.nodes:
            return "No experiments conducted yet."

        prompt = {
            "Introduction": (
                "You are an AI researcher summarizing experimental progress. "
                "Only deterministically verified experiments count as scientific "
                "successes. Unverified runnable experiments are advisory and must not "
                "support performance or causal claims."
            ),
            "Deterministically Verified Experiments": "",
            "Unverified Runnable Experiments (advisory only)": "",
            "Failed Experiments": "",
        }

        verified_nodes = self.verified_nodes
        verified_ids = {node.id for node in verified_nodes}
        for node in verified_nodes:
            exp_info = f"Design: {node.plan}\n  "
            exp_info += f"Results: {node.analysis}\n"
            exp_info += f"Metric: {str(node.metric)}\n"
            if include_code:
                exp_info += f"Code: {node.code}\n"
            prompt["Deterministically Verified Experiments"] += exp_info

        for node in self.good_nodes:
            if node.id in verified_ids:
                continue
            advisory_info = f"Design: {node.plan}\n"
            advisory_info += (
                "Verification: unavailable; do not use for ranking or claims.\n"
            )
            if include_code:
                advisory_info += f"Code: {node.code}\n"
            prompt["Unverified Runnable Experiments (advisory only)"] += advisory_info

        for node in self.buggy_nodes:
            failure_info = f"Design: {node.plan}\n  "
            failure_info += f"Error Analysis: {node.analysis}\n"
            failure_info += f"Error Type: {node.exc_type if hasattr(node, 'exc_type') else 'Unknown'}\n"
            failure_info += f"Debug Depth: {node.debug_depth}\n"
            if include_code:
                failure_info += f"Code: {node.code}\n"
            prompt["Failed Experiments"] += failure_info

        model = model_kwargs.get("model") or os.environ.get("ZHIPU_DEFAULT_MODEL")
        if not model:
            raise ValueError("Summary generation requires an explicit model")
        summary = query(
            system_message=prompt,
            user_message=(
                "Please provide a comprehensive summary of the experimental progress that includes:\n"
                "1. Key patterns supported by deterministically verified experiments\n"
                "2. Common failure patterns and pitfalls to avoid\n"
                "3. Exploratory recommendations, with unverified evidence clearly separated"
            ),
            model=model,
            temperature=model_kwargs.get("temp", 0.3),
            max_tokens=model_kwargs.get("max_tokens"),
        )

        return summary

    def generate_summary_old(self, include_code: bool = False) -> str:
        summary = []
        for n in self.good_nodes:
            summary_part = f"Design: {n.plan}\n"
            if include_code:
                summary_part += f"Code: {n.code}\n"
            summary_part += f"Results: {n.analysis}\n"
            summary_part += f"Validation Metric: {n.metric.value}\n"
            summary.append(summary_part)
        return "\n-------------------------------\n".join(summary)

    def to_dict(self, *, artifact_base: Path | None = None):
        """Convert journal to a JSON-serializable dictionary"""
        return {
            "nodes": [node.to_dict(artifact_base=artifact_base) for node in self.nodes]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Journal":
        """Restore a journal and its parent/child relationships from a dictionary."""

        if not isinstance(data, dict):
            raise ValueError("Journal payload must be an object")
        nodes_payload = data.get("nodes", [])
        if not isinstance(nodes_payload, list):
            raise ValueError("Journal nodes must be a list")

        parent_ids: dict[str, str | None] = {}
        node_payloads: list[dict] = []
        for index, node_data in enumerate(nodes_payload):
            if not isinstance(node_data, dict):
                raise ValueError(f"Journal node {index} must be an object")
            node_id = node_data.get("id")
            _validate_node_id(node_id, context=f"Journal node {index}")
            if node_id in parent_ids:
                raise ValueError(f"Journal contains duplicate node id: {node_id}")
            parent_id = node_data.get("parent_id")
            if parent_id is not None:
                try:
                    _validate_node_id(parent_id)
                except ValueError:
                    raise ValueError(
                        f"Journal node {node_id} has an invalid parent id"
                    ) from None
            parent_ids[node_id] = parent_id
            node_payloads.append(copy.deepcopy(node_data))

        node_ids = set(parent_ids)
        for node_id, parent_id in parent_ids.items():
            if parent_id is not None and parent_id not in node_ids:
                raise ValueError(
                    f"Journal node {node_id} references missing parent {parent_id}"
                )

        visited: set[str] = set()
        for start_node_id in parent_ids:
            if start_node_id in visited:
                continue
            ancestry: list[str] = []
            ancestry_positions: dict[str, int] = {}
            node_id: str | None = start_node_id
            while node_id is not None and node_id not in visited:
                if node_id in ancestry_positions:
                    raise ValueError(f"Journal parent cycle detected at node {node_id}")
                ancestry_positions[node_id] = len(ancestry)
                ancestry.append(node_id)
                node_id = parent_ids[node_id]
            visited.update(ancestry)

        journal = cls()
        for payload in node_payloads:
            node = Node.from_dict(payload, journal=None)
            node.children = set()
            journal.nodes.append(node)

        nodes_by_id = {node.id: node for node in journal.nodes}
        for node in journal.nodes:
            parent_id = parent_ids.get(node.id)
            if parent_id is not None:
                node.parent = nodes_by_id[parent_id]
                node.parent.children.add(node)
        return journal

    def save_experiment_notes(
        self, workspace_dir: str, stage_name: str, cfg: Any
    ) -> None:
        """Save experimental notes and summaries to files"""
        model, temperature, max_tokens = _experiment_notes_summary_route(cfg)
        notes_dir = os.path.join(workspace_dir, "experiment_notes")
        os.makedirs(notes_dir, exist_ok=True)

        # Get all node summaries once
        node_summaries = []
        for node in self.nodes:
            if hasattr(node, "_agent"):
                summary = node._agent._generate_node_summary(node)
                node_summaries.append(
                    {
                        "node_id": node.id,
                        "metric": str(node.metric) if node.metric else "Failed",
                        "summary": summary,
                    }
                )
                # Save individual node summary
                atomic_write_json(
                    os.path.join(
                        notes_dir, f"{stage_name}_node_{node.id}_summary.json"
                    ),
                    summary,
                    indent=2,
                    ensure_ascii=True,
                )

        summary_prompt = {
            "Introduction": "Synthesize the experimental findings from this stage",
            "Node Summaries": node_summaries,
            "Best Node": (
                {
                    "id": self.get_best_node_by_metric().id,
                    "metric": str(self.get_best_node_by_metric().metric),
                }
                if self.get_best_node_by_metric()
                else None
            ),
        }

        stage_summary = query(
            system_message=summary_prompt,
            user_message="Generate a comprehensive summary of the experimental findings in this stage",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        atomic_write_text(
            os.path.join(notes_dir, f"{stage_name}_summary.txt"), stage_summary
        )
