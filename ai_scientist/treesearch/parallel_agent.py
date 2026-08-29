from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, wait
from collections.abc import Sequence
from typing import List, Optional, Any, Callable, cast, Dict, Tuple
import ast
import hashlib
import json
import random
import re
import subprocess
import os
import stat
import unicodedata
from queue import Queue
import logging
import humanize
from .backend import (
    FunctionCallValidationError,
    FunctionSpec,
    ResearchDecisionError,
    compile_prompt_to_md,
    query,
)
from .interpreter import ExecutionResult, sandbox_policy_from_config
from .journal import Journal, Node
from .errors import ExperimentCannotContinueError
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_single_plan_and_code, wrap_code
from .utils.serialize import atomic_write_text
from ai_scientist.protocol import capture_llm_calls
from ai_scientist.utils.deterministic_evaluator import (
    DEFAULT_MAX_FILE_BYTES,
    evaluate_experiment_data,
)
from ai_scientist.utils.atomic_io import atomic_write_bytes
from ai_scientist.utils.llm_budget import is_llm_budget_exception
import copy
import dataclasses
import pickle
from dataclasses import asdict
from omegaconf import OmegaConf

from rich import print
from pathlib import Path
import base64
import sys

logger = logging.getLogger("ai-scientist")
MIN_SCIENTIFIC_SEEDS = 3
MAX_SCIENTIFIC_SEEDS = 32
MAX_RESEARCH_DECISION_RETRIES = 3
MAX_PLOT_FILE_BYTES = 100 * 1024 * 1024
MAX_PARALLEL_WORKERS = 64
SUPPORTED_DETERMINISTIC_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "rmse",
    "mse",
    "mae",
    "r2",
)

ExecCallbackType = Callable[[str, bool], ExecutionResult]


def _opaque_content_ref(value: Any) -> str:
    payload = str(value).encode("utf-8", errors="replace")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


ablation_idea_spec = FunctionSpec(
    name="propose_ablation",
    description="Propose one falsifiable component-removal experiment",
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 128},
            "component": {"type": "string", "minLength": 1, "maxLength": 256},
            "description": {"type": "string", "minLength": 1, "maxLength": 2000},
            "expected_outcome": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1000,
            },
        },
        "required": ["name", "component", "description", "expected_outcome"],
        "additionalProperties": False,
    },
)

hyperparam_idea_spec = FunctionSpec(
    name="propose_hyperparameter_trial",
    description="Propose one bounded hyperparameter change to the locked control",
    json_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1, "maxLength": 128},
            "description": {
                "type": "string",
                "minLength": 1,
                "maxLength": 2000,
            },
        },
        "required": ["name", "description"],
        "additionalProperties": False,
    },
)

metric_selection_spec = FunctionSpec(
    name="select_deterministic_metric",
    description="Select one host-supported primary scientific metric",
    json_schema={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "enum": list(SUPPORTED_DETERMINISTIC_METRICS),
            },
            "rationale": {"type": "string", "minLength": 1, "maxLength": 1000},
        },
        "required": ["metric", "rationale"],
        "additionalProperties": False,
    },
)


def _validate_retry_count(retries: int) -> int:
    if (
        isinstance(retries, bool)
        or not isinstance(retries, int)
        or not 1 <= retries <= MAX_RESEARCH_DECISION_RETRIES
    ):
        raise ValueError("retries must be an integer between 1 and 3")
    return retries


def _ablation_code_diff_hash(control_code: str, ablation_code: str) -> str:
    payload = json.dumps(
        {
            "control_code": control_code.strip(),
            "ablation_code": ablation_code.strip(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


class _ScientificSemanticNormalizer(ast.NodeTransformer):
    """Remove formatting, comments, docstrings, and statically dead branches."""

    def visit_Expr(self, node: ast.Expr):  # noqa: N802 - ast visitor API
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_If(self, node: ast.If):  # noqa: N802 - ast visitor API
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if bool(node.test.value) else node.orelse
        return node


def _semantic_code_hash(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise ExperimentCannotContinueError(
            "Experiment code has no valid semantic identity"
        ) from None
    normalized = _ScientificSemanticNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    payload = ast.dump(normalized, include_attributes=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _ablation_component_was_transformed(
    control_code: str,
    ablation_code: str,
    component: str,
) -> bool:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(component))
        if len(token) >= 3
        and token.lower()
        not in {"and", "component", "disable", "feature", "model", "remove", "the"}
    }
    if not tokens:
        return False

    def signatures(code: str) -> dict[str, set[str]]:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        normalized = _ScientificSemanticNormalizer().visit(tree)
        ast.fix_missing_locations(normalized)
        result = {token: set() for token in tokens}
        for statement in ast.walk(normalized):
            if not isinstance(statement, ast.stmt):
                continue
            identifiers: set[str] = set()
            for item in ast.walk(statement):
                if isinstance(item, ast.Name):
                    identifiers.update(item.id.lower().split("_"))
                    identifiers.add(item.id.lower())
                elif isinstance(item, ast.Attribute):
                    identifiers.update(item.attr.lower().split("_"))
                    identifiers.add(item.attr.lower())
            dumped = ast.dump(statement, include_attributes=False)
            for token in tokens & identifiers:
                result[token].add(dumped)
        return result

    control = signatures(control_code)
    ablation = signatures(ablation_code)
    relevant = [token for token in tokens if control.get(token)]
    return bool(relevant) and any(
        control[token] != ablation.get(token) for token in relevant
    )


def _canonical_idea_key(value: Any) -> str:
    """Normalize an Agent-authored idea label for durable duplicate checks."""

    if value is None:
        return ""
    folded = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = "".join(char if char.isalnum() else " " for char in folded).strip()
    return " ".join(normalized.split())


def _ablation_idea_key(name: Any, component: Any) -> str:
    return _canonical_idea_key(component)


def _configured_multi_seed_values(seed_cfg: Any) -> list[int]:
    """Resolve explicit seed sequences, including OmegaConf ListConfig values."""

    configured = getattr(seed_cfg, "seeds", None)
    if configured is not None:
        if isinstance(configured, (str, bytes)) or not isinstance(configured, Sequence):
            raise ExperimentCannotContinueError(
                "Multi-seed configuration is not a sequence"
            )
        if not MIN_SCIENTIFIC_SEEDS <= len(configured) <= MAX_SCIENTIFIC_SEEDS:
            raise ExperimentCannotContinueError(
                "Multi-seed configuration requires 3-32 values"
            )
        seeds = list(configured)
    else:
        count = getattr(seed_cfg, "num_seeds", None)
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not MIN_SCIENTIFIC_SEEDS <= count <= MAX_SCIENTIFIC_SEEDS
        ):
            raise ExperimentCannotContinueError(
                "Multi-seed count must be an integer between 3 and 32"
            )
        seeds = list(range(count))
    if any(
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
        for seed in seeds
    ):
        raise ExperimentCannotContinueError(
            "Multi-seed values must be bounded integers"
        )
    if len(set(seeds)) != len(seeds):
        raise ExperimentCannotContinueError(
            "Multi-seed configuration values must be unique"
        )
    return seeds


def _inject_seed_bootstrap(code: str, seed: int) -> tuple[str, str]:
    """Rewrite one explicit training RNG seed while preserving the data seed."""

    if (
        not isinstance(code, str)
        or not code.strip()
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= 2**32 - 1
    ):
        raise ExperimentCannotContinueError("Seed injection inputs are invalid")
    try:
        tree = ast.parse(code)
    except SyntaxError:
        raise ExperimentCannotContinueError(
            "Qualified experiment code is not valid Python"
        ) from None
    seed_names = {"XSCIENTIST_DATA_SEED", "XSCIENTIST_TRAINING_SEED"}
    assignments: dict[str, ast.Assign | ast.AnnAssign] = {}
    allowed_store_ids: set[int] = set()
    for statement in tree.body:
        target: ast.Name | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target = statement.target
        if target is not None and target.id in seed_names:
            if target.id in assignments:
                raise ExperimentCannotContinueError(
                    "Experiment seed roles must have one immutable declaration"
                )
            assignments[target.id] = statement
            allowed_store_ids.add(id(target))

    forbidden_binding = False
    for item in ast.walk(tree):
        if (
            isinstance(item, ast.Name)
            and item.id in seed_names
            and isinstance(item.ctx, (ast.Store, ast.Del))
            and id(item) not in allowed_store_ids
        ):
            forbidden_binding = True
        if isinstance(item, ast.arg) and item.arg in seed_names:
            forbidden_binding = True
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            forbidden_binding |= item.name in seed_names
        if isinstance(item, ast.alias):
            bound_name = item.asname or item.name.split(".", 1)[0]
            forbidden_binding |= bound_name in seed_names
        if isinstance(item, ast.ExceptHandler):
            forbidden_binding |= item.name in seed_names
        if isinstance(item, (ast.Global, ast.Nonlocal)):
            forbidden_binding |= any(name in seed_names for name in item.names)
        if isinstance(item, (ast.MatchAs, ast.MatchStar)):
            forbidden_binding |= item.name in seed_names
        if isinstance(item, ast.MatchMapping):
            forbidden_binding |= item.rest in seed_names
    if set(assignments) != seed_names or forbidden_binding:
        raise ExperimentCannotContinueError(
            "Experiment code must declare two distinct immutable seed roles"
        )

    for name, assignment in assignments.items():
        value = assignment.value
        if (
            not isinstance(value, ast.Constant)
            or isinstance(value.value, bool)
            or not isinstance(value.value, int)
            or not 0 <= value.value <= 2**32 - 1
        ):
            raise ExperimentCannotContinueError(
                f"{name} must be assigned a bounded integer literal"
            )

    import_aliases: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                import_aliases[alias.asname or alias.name.split(".", 1)[0]] = alias.name
        elif isinstance(statement, ast.ImportFrom) and statement.module:
            for alias in statement.names:
                if alias.name != "*":
                    import_aliases[alias.asname or alias.name] = (
                        f"{statement.module}.{alias.name}"
                    )

    alias_names = set(import_aliases)
    for item in ast.walk(tree):
        rebound_name = None
        if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            rebound_name = item.id
        elif isinstance(item, ast.arg):
            rebound_name = item.arg
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            rebound_name = item.name
        if rebound_name in alias_names:
            raise ExperimentCannotContinueError(
                "RNG import aliases must not be rebound"
            )

    recognized_seed_calls = {
        "jax.random.PRNGKey",
        "numpy.random.RandomState",
        "numpy.random.SeedSequence",
        "numpy.random.default_rng",
        "numpy.random.seed",
        "random.Random",
        "random.seed",
        "tensorflow.random.set_seed",
        "torch.cuda.manual_seed_all",
        "torch.manual_seed",
    }

    def qualified_call_name(function: ast.AST) -> str:
        if isinstance(function, ast.Name):
            return import_aliases.get(function.id, function.id)
        if isinstance(function, ast.Attribute):
            prefix = qualified_call_name(function.value)
            resolved = f"{prefix}.{function.attr}" if prefix else function.attr
            root, separator, remainder = resolved.partition(".")
            if separator and root in import_aliases:
                return f"{import_aliases[root]}.{remainder}"
            return resolved
        return ""

    def seed_argument(call: ast.Call) -> ast.AST | None:
        if call.args:
            return call.args[0]
        keyword_names = {
            "jax.random.PRNGKey": {"seed"},
            "numpy.random.RandomState": {"seed"},
            "numpy.random.SeedSequence": {"entropy"},
            "numpy.random.default_rng": {"seed"},
            "numpy.random.seed": {"seed"},
            "random.Random": {"x"},
            "random.seed": {"a"},
            "tensorflow.random.set_seed": {"seed"},
            "torch.cuda.manual_seed_all": {"seed"},
            "torch.manual_seed": {"seed"},
        }[qualified_call_name(call.func)]
        matching = [kw.value for kw in call.keywords if kw.arg in keyword_names]
        return matching[0] if len(matching) == 1 else None

    def direct_seed_role(call: ast.Call) -> str | None:
        if qualified_call_name(call.func) not in recognized_seed_calls:
            return None
        argument = seed_argument(call)
        if (
            isinstance(argument, ast.Name)
            and isinstance(argument.ctx, ast.Load)
            and argument.id in seed_names
        ):
            return argument.id
        return ""

    local_functions = {
        statement.name: statement
        for statement in tree.body
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    class _ReachableCallCollector(ast.NodeVisitor):
        """Collect module-reachable calls without trusting dead helper bodies."""

        def __init__(self) -> None:
            self.calls: list[ast.Call] = []
            self._active_functions: set[str] = set()
            self._conditional_depth = 0
            self.conditional_seed_call = False

        def _visit_conditionally(self, nodes: list[ast.AST]) -> None:
            self._conditional_depth += 1
            try:
                self._visit_block(nodes)
            finally:
                self._conditional_depth -= 1

        def _visit_block(self, nodes: list[ast.AST]) -> None:
            conditionally_reachable = False
            for child in nodes:
                if conditionally_reachable:
                    self._visit_conditionally([child])
                else:
                    self.visit(child)
                if isinstance(child, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                    break
                if self._can_interrupt_flow(child):
                    conditionally_reachable = True

        @classmethod
        def _can_interrupt_flow(cls, node: ast.AST) -> bool:
            """Conservatively detect paths that can skip following statements."""

            if isinstance(node, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                return True
            if isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda),
            ):
                return False
            if isinstance(node, ast.If) and isinstance(node.test, ast.Constant):
                selected = node.body if bool(node.test.value) else node.orelse
                return any(cls._can_interrupt_flow(item) for item in selected)
            return any(
                cls._can_interrupt_flow(child) for child in ast.iter_child_nodes(node)
            )

        @staticmethod
        def _is_main_guard(test: ast.AST) -> bool:
            return (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            )

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            return None

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            return None

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            return None

        def visit_If(self, node: ast.If) -> None:
            if isinstance(node.test, ast.Constant):
                selected = node.body if bool(node.test.value) else node.orelse
                self._visit_block(list(selected))
                return
            if self._is_main_guard(node.test):
                self._visit_block(list(node.body))
                return
            self.visit(node.test)
            self._visit_conditionally(list(node.body))
            self._visit_conditionally(list(node.orelse))

        def visit_While(self, node: ast.While) -> None:
            if isinstance(node.test, ast.Constant) and not bool(node.test.value):
                self._visit_block(list(node.orelse))
                return
            self.visit(node.test)
            self._visit_conditionally(list(node.body))
            self._visit_conditionally(list(node.orelse))

        def visit_For(self, node: ast.For) -> None:
            self.visit(node.iter)
            self._visit_conditionally(list(node.body))
            self._visit_conditionally(list(node.orelse))

        def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
            self.visit(node.iter)
            self._visit_conditionally(list(node.body))
            self._visit_conditionally(list(node.orelse))

        def visit_IfExp(self, node: ast.IfExp) -> None:
            self.visit(node.test)
            self._visit_conditionally([node.body, node.orelse])

        def visit_BoolOp(self, node: ast.BoolOp) -> None:
            if node.values:
                self.visit(node.values[0])
                self._visit_conditionally(list(node.values[1:]))

        def visit_Try(self, node: ast.Try) -> None:
            self._visit_conditionally(list(node.body))
            self._visit_conditionally(list(node.orelse))
            self._visit_conditionally(list(node.finalbody))
            for handler in node.handlers:
                self._visit_conditionally(list(handler.body))

        def visit_Lambda(self, node: ast.Lambda) -> None:
            return None

        def visit_Call(self, node: ast.Call) -> None:
            self.calls.append(node)
            if (
                self._conditional_depth
                and qualified_call_name(node.func) in recognized_seed_calls
            ):
                self.conditional_seed_call = True
            if isinstance(node.func, ast.Name) and node.func.id in local_functions:
                name = node.func.id
                if name not in self._active_functions:
                    self._active_functions.add(name)
                    self._visit_block(list(local_functions[name].body))
                    self._active_functions.remove(name)
            self.generic_visit(node)

    collector = _ReachableCallCollector()
    collector._visit_block(
        [
            statement
            for statement in tree.body
            if not isinstance(
                statement,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
            )
        ]
    )
    calls = collector.calls
    reachable_seed_calls = [
        call
        for call in calls
        if qualified_call_name(call.func) in recognized_seed_calls
    ]
    applied_roles = [direct_seed_role(call) for call in reachable_seed_calls]
    if (
        collector.conditional_seed_call
        or "" in applied_roles
        or set(applied_roles) != seed_names
    ):
        raise ExperimentCannotContinueError(
            "Every RNG seed call must unconditionally use one declared seed role"
        )

    training_assignment = assignments["XSCIENTIST_TRAINING_SEED"]
    training_assignment.value = ast.Constant(value=seed)
    ast.fix_missing_locations(tree)
    seeded_code = ast.unparse(tree) + "\n"
    try:
        ast.parse(seeded_code)
    except SyntaxError:
        raise ExperimentCannotContinueError(
            "Seeded experiment code is invalid"
        ) from None
    receipt_payload = {
        "schema": "xscientist.seed-bootstrap.v3",
        "method": "reachable_direct_rng_seed_role_v3",
        "seed_application_syntax_verified": True,
        "seed": seed,
        "parent_code_sha256": "sha256:"
        + hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "seeded_code_sha256": "sha256:"
        + hashlib.sha256(seeded_code.encode("utf-8")).hexdigest(),
    }
    receipt = json.dumps(
        receipt_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return seeded_code, "sha256:" + hashlib.sha256(receipt).hexdigest()


def _validate_confirmation_seed_set(code: str, seeds: Sequence[int]) -> None:
    """Require confirmation seeds to be held out from the selection run."""

    try:
        tree = ast.parse(code)
    except (SyntaxError, TypeError):
        raise ExperimentCannotContinueError(
            "Qualified experiment code is not valid Python"
        ) from None
    training_seeds: list[int] = []
    for statement in tree.body:
        target: ast.Name | None = None
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign) and isinstance(
            statement.target, ast.Name
        ):
            target = statement.target
        if target is not None and target.id == "XSCIENTIST_TRAINING_SEED":
            value = statement.value
            if (
                not isinstance(value, ast.Constant)
                or isinstance(value.value, bool)
                or not isinstance(value.value, int)
            ):
                raise ExperimentCannotContinueError(
                    "XSCIENTIST_TRAINING_SEED must be an integer literal"
                )
            training_seeds.append(value.value)
    if len(training_seeds) != 1:
        raise ExperimentCannotContinueError(
            "Experiment code must declare one training seed"
        )
    if training_seeds[0] in seeds:
        raise ExperimentCannotContinueError(
            "Confirmation seeds must be held out from the selection training seed"
        )


def _publish_source_artifacts(
    output_dir: str | Path, artifacts: Dict[str, str]
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in artifacts.items():
        atomic_write_text(output_dir / filename, content)


def _preserve_evaluation_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    expected_hash: str,
) -> Path:
    """Copy the exact evaluated bytes into durable, replayable evidence."""

    payload = _read_untrusted_regular_file(
        source,
        max_bytes=DEFAULT_MAX_FILE_BYTES,
        purpose="evaluation artifact",
    )
    actual_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
    if actual_hash != expected_hash:
        raise ExperimentCannotContinueError(
            "Evaluation artifact no longer matches its evaluator receipt"
        )
    atomic_write_bytes(destination, payload)
    return Path(destination)


def _read_untrusted_regular_file(
    source: str | Path,
    *,
    max_bytes: int,
    purpose: str,
) -> bytes:
    """Read a bounded regular file without following an Agent-created symlink."""

    source = Path(source)
    if source.is_symlink():
        raise ExperimentCannotContinueError(f"Unsafe {purpose} symlink")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError:
        raise ExperimentCannotContinueError(f"Cannot safely open {purpose}") from None
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 0
            or before.st_size > max_bytes
        ):
            raise ExperimentCannotContinueError(f"Unsafe or oversized {purpose}")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ExperimentCannotContinueError(f"{purpose} changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExperimentCannotContinueError(f"{purpose} changed while reading")
    return b"".join(chunks)


def _copy_untrusted_png(source: str | Path, destination: str | Path) -> Path:
    payload = _read_untrusted_regular_file(
        source,
        max_bytes=MAX_PLOT_FILE_BYTES,
        purpose="plot artifact",
    )
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ExperimentCannotContinueError("Plot artifact is not a PNG file")
    atomic_write_bytes(destination, payload)
    return Path(destination)


def _publish_plot_artifacts(
    source_dir: str | Path,
    destination_dir: str | Path,
) -> list[Path]:
    """Publish PNGs only; plotting arrays can never replace scientific evidence."""

    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    published: list[Path] = []
    for source in sorted(source_dir.glob("*.png")):
        destination = destination_dir / source.name
        _copy_untrusted_png(source, destination)
        published.append(destination)
    return published


def _assert_preserved_evaluation_artifact(node: Node) -> None:
    if not node.has_verified_metric:
        return
    evidence_dir = Path(node.exp_results_dir)
    artifact = evidence_dir / "experiment_data.npy"
    code_artifact = evidence_dir / "experiment_code.py"
    if artifact.is_symlink() or code_artifact.is_symlink():
        raise ExperimentCannotContinueError("Preserved evidence is unsafe")
    try:
        preserved_code = code_artifact.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise ExperimentCannotContinueError(
            "Preserved evidence is unreadable"
        ) from None
    replay = evaluate_experiment_data(
        artifact,
        requested_metric=node.evaluation_report.get("requested_metric"),
    )
    if preserved_code != node.code or replay != node.evaluation_report:
        raise ExperimentCannotContinueError(
            "Preserved scientific evidence changed after evaluation"
        )


def _sandbox_policy_for_workspace(cfg: Config):
    """Expose only explicit read-only roots needed by experiment code."""

    policy = sandbox_policy_from_config(cfg.exec)
    read_only_mounts = set(policy.read_only_mounts)
    data_dir = getattr(cfg, "data_dir", None)
    if data_dir:
        read_only_mounts.add(str(Path(data_dir).resolve()))
    log_dir = getattr(cfg, "log_dir", None)
    if log_dir:
        read_only_mounts.add(str(Path(log_dir).resolve()))
    return dataclasses.replace(
        policy,
        read_only_mounts=tuple(sorted(read_only_mounts)),
    )


def _experiment_execution_env() -> dict[str, str | None]:
    allowed = ("CUDA_VISIBLE_DEVICES",)
    return {name: os.getenv(name) for name in allowed}


def _interpreter_for_workspace(
    cfg: Config,
    workspace: str | Path,
    *,
    allow_network: bool,
):
    from .interpreter import Interpreter

    policy = _sandbox_policy_for_workspace(cfg)
    if allow_network and policy.network == "none":
        policy = dataclasses.replace(policy, network="bridge")
    return Interpreter(
        working_dir=workspace,
        timeout=cfg.exec.timeout,
        format_tb_ipython=cfg.exec.format_tb_ipython,
        agent_file_name=cfg.exec.agent_file_name,
        env_vars=_experiment_execution_env(),
        sandbox_policy=policy,
    )


def _experiment_network_enabled(cfg: Config) -> bool:
    requested = bool(getattr(cfg.exec, "allow_experiment_network", False))
    if requested and bool(getattr(cfg.exec, "require_isolation", False)):
        raise ValueError(
            "Strict isolation cannot be combined with experiment network access; "
            "set exec.allow_experiment_network=false and pre-cache inputs."
        )
    return requested


def _safe_pickle_test(obj, name="object"):
    """Test if an object can be pickled"""
    try:
        pickle.dumps(obj)
        return True
    except Exception as e:
        logger.error("Cannot pickle %s: %s", name, type(e).__name__)
        return False


def _parse_keyword_prefix_response(
    response: str, keyword_prefix1: str, keyword_prefix2: str
) -> Tuple[Optional[str], Optional[str]]:
    """Parse the response into name and description based on keyword prefix"""
    try:
        # Split response into lines and clean up
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        # Find the idea and description
        name = None
        description = None

        for line in lines:
            if line.startswith(keyword_prefix1):
                name = line.replace(keyword_prefix1, "").strip()
            elif line.startswith(keyword_prefix2):
                description = line.replace(keyword_prefix2, "").strip()
                # Combine any following lines that don't start with a marker
                desc_lines = []
                for next_line in lines[lines.index(line) + 1 :]:
                    if not next_line.startswith((keyword_prefix1, keyword_prefix2)):
                        desc_lines.append(next_line)
                    else:
                        break
                if desc_lines:
                    description = " ".join([description] + desc_lines)

        if name is None or description is None:
            raise ValueError(
                f"Missing required keywords in response: {keyword_prefix1} and/or {keyword_prefix2}"
            )

        return name, description

    except Exception as e:
        logger.error("Error parsing research response: %s", type(e).__name__)
        logger.debug("Unparsed research response length: %d", len(response))
        return None, None


def _extract_dataset_names_from_metric(metric: Optional[MetricValue]) -> List[str]:
    """
    Extract dataset names from a MetricValue produced by metric parsing.

    This is more stable than inferring from plot analysis, and is used for stage
    completion gating (e.g. "tested on >= N datasets").
    """
    if metric is None or metric.value is None:
        return []

    names = set()
    value = metric.value
    if isinstance(value, dict):
        if "metric_names" in value:
            for entry in value.get("metric_names", []) or []:
                for point in entry.get("data", []) or []:
                    ds = point.get("dataset_name")
                    if ds:
                        names.add(str(ds).strip())
        else:
            for key in value.keys():
                if key:
                    names.add(str(key).strip())

    return sorted(name for name in names if name)


review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, summarize the bug and propose a fix. Otherwise, leave it empty.",
            },
        },
        "required": [
            "is_bug",
            "summary",
        ],
        "additionalProperties": False,
    },
    description="Submit a review evaluating the output of the training script.",
)

vlm_feedback_spec = FunctionSpec(
    name="analyze_experiment_plots",
    json_schema={
        "type": "object",
        "properties": {
            "plot_analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "Detailed analysis of the plot's results and implications",
                        },
                    },
                    "required": ["analysis"],
                    "additionalProperties": False,
                },
            },
            "valid_plots_received": {
                "type": "boolean",
                "description": "True if valid plots were received, False otherwise. For example, if the plots are empty or not meaningful, this should be False.",
            },
            "vlm_feedback_summary": {
                "type": "string",
                "description": "Summarize the feedback from the VLM. If the task involves generative modeling, make sure to focus on the generated samples.",
            },
        },
        "required": ["plot_analyses", "valid_plots_received", "vlm_feedback_summary"],
        "additionalProperties": False,
    },
    description="Analyze experimental plots and provide detailed feedback on the results.",
)

metric_parse_spec = FunctionSpec(
    name="parse_metrics",
    json_schema={
        "type": "object",
        "properties": {
            "valid_metrics_received": {
                "type": "boolean",
                "description": "True if the metrics were successfully received, False otherwise. For example if the execution output does not contain any metrics, set this to False.",
            },
            "metric_names": {
                "type": "array",
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "metric_name": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 256,
                            "description": "Specify the metric name clearly. Avoid vague terms like 'train,' 'val,' or 'test.' Instead, use precise labels such as 'train accuracy,' 'validation loss,' or 'test F1 score,' etc.",
                        },
                        "lower_is_better": {
                            "type": "boolean",
                            "description": "Whether lower values are better for this metric",
                        },
                        "description": {
                            "type": "string",
                            "maxLength": 4096,
                            "description": "Description of the metric",
                        },
                        "data": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 256,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "dataset_name": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 256,
                                        "description": "The name of the dataset. Never include 'train', 'val', or 'test' in the dataset name.",
                                    },
                                    "final_value": {
                                        "type": "number",
                                        "description": "The final value of the metric for this dataset",
                                    },
                                    "best_value": {
                                        "type": "number",
                                        "description": "The best value of the metric for this dataset",
                                    },
                                },
                                "required": [
                                    "dataset_name",
                                    "final_value",
                                    "best_value",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "data",
                        "metric_name",
                        "lower_is_better",
                        "description",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["valid_metrics_received", "metric_names"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {
                    "properties": {"valid_metrics_received": {"const": True}},
                    "required": ["valid_metrics_received"],
                    "additionalProperties": True,
                },
                "then": {
                    "properties": {"metric_names": {"minItems": 1, "maxItems": 1}},
                    "additionalProperties": True,
                },
                "else": {
                    "properties": {"metric_names": {"maxItems": 0}},
                    "additionalProperties": True,
                },
            }
        ],
    },
    description="Parse metrics from execution output",
)


plot_selection_spec = FunctionSpec(
    name="select_plots",
    json_schema={
        "type": "object",
        "properties": {
            "selected_plots": {
                "type": "array",
                "description": "List of selected plot file paths",
                "items": {"type": "string", "description": "Full path to a plot file"},
                "minItems": 1,
                "maxItems": 10,
                "uniqueItems": True,
            }
        },
        "required": ["selected_plots"],
        "additionalProperties": False,
    },
    description="Select the 10 most relevant plots for analysis",
)

experiment_summary_spec = FunctionSpec(
    name="summarize_experiment",
    description="Summarize experimental findings",
    json_schema={
        "type": "object",
        "properties": {
            "findings": {
                "type": "string",
                "description": "Key findings and results",
            },
            "significance": {
                "type": "string",
                "description": "Why these results matter",
            },
            "next_steps": {
                "type": "string",
                "description": "Suggested improvements or next experiments",
            },
        },
        "required": ["findings", "significance"],
        "additionalProperties": False,
    },
)


class AblationConfig:
    """Track state of ablation experiments"""

    def __init__(self, name: str, description: str, code: str, base_node: Node):
        self.name = name
        self.description = description
        self.code = code
        self.base_node = base_node
        self.attempts = 0
        self.max_attempts = 3  # Maximum number of retry attempts
        self.last_error = None
        self.completed = False
        self.current_node = None


class AblationIdea:
    """Ablation idea"""

    def __init__(
        self,
        name: str,
        description: str,
        component: str,
        expected_outcome: str,
        llm_call_refs: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.component = component
        self.expected_outcome = expected_outcome
        self.llm_call_refs = list(llm_call_refs or [])


class HyperparamTuningIdea:
    """Hyperparameter tuning idea"""

    def __init__(
        self,
        name: str,
        description: str,
        llm_call_refs: list[str] | None = None,
    ):
        self.name = name
        self.description = description
        self.llm_call_refs = list(llm_call_refs or [])


class MinimalAgent:
    """A minimal agent class that only contains what's needed for processing nodes"""

    def __init__(
        self,
        task_desc,
        cfg,
        memory_summary=None,
        evaluation_metrics=None,
        stage=None,
        stage_name=None,
    ):
        self.task_desc = task_desc
        self.memory_summary = memory_summary
        self.cfg = cfg
        self.evaluation_metrics = evaluation_metrics
        self.stage_name = stage_name
        self.data_preview = None

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "scikit-learn",
            "statsmodels",
            "xgboost",
            "lightGBM",
            "torch",
            "torchvision",
            "torch-geometric",
            "bayesian-optimization",
            "timm",
            "albumentations",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant machine learning packages such as: {pkg_str}. Feel free to use any other packages too (all packages are already installed!). For neural networks we suggest using PyTorch rather than TensorFlow."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "CRITICAL GPU REQUIREMENTS - Your code MUST include ALL of these:",
            "  - At the start of your code, add these lines to handle GPU/CPU:",
            "    ```python",
            "    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
            "    print(f'Using device: {device}')",
            "    ```",
            "  - ALWAYS move models to device using the `.to(device)` method",
            "  - ALWAYS move input tensors to device using the `.to(device)` method",
            "  - ALWAYS move model related tensors to device using the `.to(device)` method",
            "  - For optimizers, create them AFTER moving model to device",
            "  - When using DataLoader, move batch tensors to device in training loop: `batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}`",
            "CRITICAL MODEL INPUT GUIDELINES:",
            "  - Always pay extra attention to the input to the model being properly normalized",
            "  - This is extremely important because the input to the model's forward pass directly affects the output, and the loss function is computed based on the output",
        ]
        if hasattr(self.cfg.experiment, "num_syn_datasets"):
            num_syn_datasets = self.cfg.experiment.num_syn_datasets
            if num_syn_datasets > 1:
                impl_guideline.extend(
                    [
                        f"You MUST evaluate your solution on at least {num_syn_datasets} different synthetic datasets to ensure robustness:",
                        "  - Use standard benchmark datasets when available",
                        f"  - If using synthetic data, generate at least {num_syn_datasets} variants with different characteristics",
                        "  - Report metrics separately for each dataset",
                        "  - Compute and report the average metric across all datasets",
                    ]
                )
        impl_guideline.extend(
            [
                "For generative modeling tasks, you must:",
                "  - Generate a set of samples from your model",
                "  - Compare these samples with ground truth data using appropriate visualizations",
                "  - When saving plots, always use the 'working_dir' variable that will be defined at the start of the script",
                "  - Make sure to give each figure a unique and appropriate name based on the dataset it represents, rather than reusing the same filename.",
                "Important code structure requirements:",
                "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block",
                "  - All code should be at the global scope or in functions that are called from the global scope",
                "  - The script should execute immediately when run, without requiring any special entry point",
                "The code should start with:",
                "  import os",
                "  working_dir = os.path.join(os.getcwd(), 'working')",
                "  os.makedirs(working_dir, exist_ok=True)",
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Your response should only contain a single code block.",
                f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}.",
                'You can also use the "./working" directory to store any temporary files that your code needs to create.',
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'dataset_name_1': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          'sample_ids': [],  # stable source/example IDs, unique within this split",
                "          'evaluation_inputs': [],  # exact raw inputs, one row per ground-truth record",
                "          # Add other relevant data",
                "      },",
                "      # Add additional datasets as needed:",
                "      'dataset_name_2': {",
                "          'metrics': {'train': [], 'val': []},",
                "          'losses': {'train': [], 'val': []},",
                "          'predictions': [],",
                "          'ground_truth': [],",
                "          'sample_ids': [],",
                "          'evaluation_inputs': [],",
                "          # Add other relevant data",
                "      },",
                "  }",
                "  # During training/evaluation:",
                "  experiment_data['dataset_name_1']['metrics']['train'].append(train_metric)",
                "  ```",
                "- Include timestamps or epochs with the saved metrics",
                "- For every evaluated record, save its stable source sample_id and exact raw evaluation input before preprocessing; the deterministic evaluator computes fingerprints itself, and lengths/order MUST match ground_truth",
                "- Declare top-level integer XSCIENTIST_DATA_SEED and XSCIENTIST_TRAINING_SEED variables. Use DATA_SEED only for dataset generation, sampling, and splitting; use TRAINING_SEED only for initialization, shuffling, optimization, and other training randomness",
                "- Evaluate the unchanged control and every candidate on the exact same ordered record identities and split paths",
                "- For large datasets, consider saving in chunks or using np.savez_compressed()",
                "CRITICAL EVALUATION REQUIREMENTS - Your code MUST include ALL of these:",
                "  1. Track and print validation loss at each epoch or at suitable intervals:",
                "     ```python",
                "     print(f'Epoch {{epoch}}: validation_loss = {{val_loss:.4f}}')",
                "     ```",
                "  2. Track and update ALL these additional metrics: "
                + str(self.evaluation_metrics),
                "  3. Update metrics at EACH epoch:",
                "  4. Save ALL metrics at the end:",
                "     ```python",
                "     np.save(os.path.join(working_dir, 'experiment_data.npy'), experiment_data)",
                "     ```",
            ]
        )

        if self.cfg.agent.k_fold_validation > 1:
            impl_guideline.append(
                f"The evaluation should be based on {self.cfg.agent.k_fold_validation}-fold cross-validation but only if that's an appropriate evaluation for the task at hand."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Make sure to write concise code."
            )
        }

    def _prompt_metricparse_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code for the metric parsing. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. "
            )
        }

    @property
    def _prompt_debug_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the bugfix/solution. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
                "Your generated code should be complete and executable. Do not omit any part of the code, even if it was part of a previous implementation."
                "Make sure to write concise code."
            )
        }

    @property
    def _prompt_hyperparam_tuning_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including hyperparameter tuning. "
                "There should be no additional headings or text in your response. Do not omit any part of the code, "
                "Your generated code should be complete and executable."
                "Make sure to write concise code."
            )
        }

    @property
    def _prompt_ablation_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (using the format ```python ... ```) which implements the full code including the ablation study. "
                "There should be no additional headings or text in your response. Do not omit any part of the code, "
                "Your generated code should be complete and executable."
                "Make sure to write concise code."
            )
        }

    def _draft(self) -> Node:
        # ARA seed short-circuit: if a fork manifest is staged, honour it and
        # skip the LLM call. Import lazily so BFTS keeps working when the
        # utils package isn't on sys.path (e.g. in trimmed dev environments).
        try:
            from ai_scientist.utils.ara_seed import load_active_seed
        except Exception:  # pragma: no cover - defensive
            load_active_seed = None  # type: ignore[assignment]
        seed = load_active_seed() if load_active_seed else None
        if seed:
            code = str(seed.get("code") or "")
            plan = str(seed.get("plan") or "Seed from ARA fork.")
            provenance = seed.get("provenance") or {}
            print(
                "[cyan]ARA seed detected — bypassing LLM draft. "
                f"parent_node_id={provenance.get('parent_node_id')} "
                f"parent_content_hash={provenance.get('parent_content_hash')}[/cyan]"
            )
            # Seed-derived nodes have no LLM origin, so llm_call_refs stays
            # empty. Provenance links back to the parent via the seed manifest.
            return Node(plan=plan, code=code)

        prompt: Any = {
            "Introduction": (
                "You are an AI researcher who is looking to publish a paper that will contribute significantly to the field."
                "Your first task is to write a python code to implement a solid baseline based on your research idea provided below, "
                "from data preparation to model training, as well as evaluation and visualization. "
                "Focus on getting a simple but working implementation first, before any sophisticated improvements. "
                "We will explore more advanced variations in later stages."
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Experiment design sketch guideline": [
                "This first experiment design should be relatively simple, without extensive hyper-parameter optimization.",
                "Take the Memory section into consideration when proposing the design. ",
                "The solution sketch should be 6-10 sentences. ",
                "Don't suggest to do EDA.",
                "Make sure to create synthetic data if needed.",
                "",
            ],
            "Evaluation Metric(s)": self.evaluation_metrics,
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.cfg.agent.data_preview:
            prompt["Data Overview"] = self.data_preview

        logger.info(
            "Research task loaded (ref=%s)",
            _opaque_content_ref(self.task_desc),
        )

        print("MinimalAgent: Getting plan and code")
        # Capture semantic LLM call-receipt hashes that produced this node's code.
        # When the ARA tracer is inactive the block is a no-op (refs stays []).
        with capture_llm_calls() as refs:
            plan, code = self.plan_and_code_query(prompt)
        print("MinimalAgent: Draft complete")
        return Node(plan=plan, code=code, llm_call_refs=list(refs))

    def _debug(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. Your previous code for research experiment had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Research idea": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Feedback based on generated plots": parent_node.vlm_feedback_summary,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_debug_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
                "Don't suggest to do EDA.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.cfg.agent.data_preview:
            prompt["Data Overview"] = self.data_preview

        with capture_llm_calls() as refs:
            plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node, llm_call_refs=list(refs))

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "implementation. Your task is to improve it based on the current experimental stage."
            ),
            "Research idea": self.task_desc,
            "Memory": self.memory_summary if self.memory_summary else "",
            "Feedback based on generated plots": parent_node.vlm_feedback_summary,
            "Feedback about execution time": parent_node.exec_time_feedback,
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= self._prompt_impl_guideline

        with capture_llm_calls() as refs:
            plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
            llm_call_refs=list(refs),
        )

    def _generate_seed_node(self, parent_node: Node):
        return Node(
            plan="Seed node",
            code=parent_node.code,
            parent=parent_node,
            is_seed_node=True,
            random_seed=parent_node.random_seed,
            seed_bootstrap_hash=parent_node.seed_bootstrap_hash,
            ablation_name=parent_node.ablation_name,
            ablation_control_node_id=parent_node.ablation_control_node_id,
            ablation_component=parent_node.ablation_component,
            ablation_expected_outcome=parent_node.ablation_expected_outcome,
            ablation_code_diff_hash=parent_node.ablation_code_diff_hash,
            ablation_control_semantic_hash=parent_node.ablation_control_semantic_hash,
            ablation_semantic_hash=parent_node.ablation_semantic_hash,
        )

    def _generate_hyperparam_tuning_node(
        self, parent_node: Node, hyperparam_idea: HyperparamTuningIdea
    ):
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "baseline implementation. Your task is to implement hyperparameter tuning for the following idea: "
                + hyperparam_idea.name
                + ". "
                + hyperparam_idea.description
            ),
            "Base code you are working on": wrap_code(parent_node.code),
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Implementation guideline": [
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'hyperparam_tuning_type_1': {",
                "          'dataset_name_1': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              'sample_ids': [],",
                "              'evaluation_inputs': [],",
                "              # Add other relevant data",
                "          },",
                "          # Add additional datasets as needed:",
                "      },",
                "      # Add additional hyperparam tuning types as needed",
                "  }",
                "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
                "Reuse the control's exact dataset paths, ordered sample_ids, ground_truth, and raw evaluation inputs.",
                "Preserve XSCIENTIST_DATA_SEED exactly; vary only XSCIENTIST_TRAINING_SEED for training randomness.",
            ]
        }
        prompt["Instructions"] |= self._prompt_hyperparam_tuning_resp_fmt
        with capture_llm_calls() as refs:
            plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan="Hyperparam tuning name: " + hyperparam_idea.name + ".\n" + plan,
            code=code,
            parent=parent_node,
            hyperparam_name=hyperparam_idea.name,
            llm_call_refs=[*hyperparam_idea.llm_call_refs, *refs],
        )

    def _generate_ablation_node(self, parent_node: Node, ablation_idea: AblationIdea):
        prompt: Any = {
            "Introduction": (
                "You are an experienced AI researcher. You are provided with a previously developed "
                "baseline implementation. Your task is to implement the ablation study for the following idea: "
                + ablation_idea.name
                + ". "
                + ablation_idea.description
                + " Remove or disable exactly this component: "
                + ablation_idea.component
                + ". Expected discriminating outcome: "
                + ablation_idea.expected_outcome
            ),
            "Base code you are working on": wrap_code(parent_node.code),
            "Instructions": {},
        }
        prompt["Instructions"] |= {
            "Implementation guideline": [
                "The code should be a single-file python program that is self-contained and can be executed as-is.",
                "No parts of the code should be skipped, don't terminate the code execution before finishing the script.",
                "Data saving requirements:",
                "- Save all plottable data (metrics, losses, predictions, etc.) as numpy arrays using np.save()",
                "- Use the following naming convention for saved files:",
                "  ```python",
                "  # At the start of your code",
                "  experiment_data = {",
                "      'ablation_type_1': {",
                "          'dataset_name_1': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              'sample_ids': [],",
                "              'evaluation_inputs': [],",
                "              # Add other relevant data",
                "          },",
                "          # Add additional datasets as needed:",
                "          'dataset_name_2': {",
                "              'metrics': {'train': [], 'val': []},",
                "              'losses': {'train': [], 'val': []},",
                "              'predictions': [],",
                "              'ground_truth': [],",
                "              'sample_ids': [],",
                "              'evaluation_inputs': [],",
                "              # Add other relevant data",
                "          },",
                "      },",
                "      # Add additional ablation types as needed",
                "  }",
                "Make sure to use a filename 'experiment_data.npy' to save the data. Do not use any other filename.",
                "Reuse the control's exact dataset paths, ordered sample_ids, ground_truth, and raw evaluation inputs.",
                "Preserve XSCIENTIST_DATA_SEED exactly; vary only XSCIENTIST_TRAINING_SEED for training randomness.",
            ]
        }
        prompt["Instructions"] |= self._prompt_ablation_resp_fmt
        with capture_llm_calls() as refs:
            plan, code = self.plan_and_code_query(prompt)
        control_semantic_hash = _semantic_code_hash(parent_node.code)
        ablation_semantic_hash = _semantic_code_hash(code)
        return Node(
            plan="Ablation name: " + ablation_idea.name + ".\n" + plan,
            code=code,
            parent=parent_node,
            ablation_name=ablation_idea.name,
            ablation_control_node_id=parent_node.id,
            ablation_component=ablation_idea.component,
            ablation_expected_outcome=ablation_idea.expected_outcome,
            ablation_code_diff_hash=_ablation_code_diff_hash(parent_node.code, code),
            ablation_control_semantic_hash=control_semantic_hash,
            ablation_semantic_hash=ablation_semantic_hash,
            llm_call_refs=[*ablation_idea.llm_call_refs, *refs],
        )

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        retries = _validate_retry_count(retries)
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            try:
                return extract_single_plan_and_code(completion_text)
            except ValueError:
                pass

            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        raise ResearchDecisionError(
            "Research Agent returned malformed plan/code after bounded retries"
        )

    def parse_exec_result(
        self, node: Node, exec_result: ExecutionResult, workspace: str
    ):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are an experienced AI researcher. "
                "You have written code for your research experiment and now need to evaluate the output of the code execution. "
                "Analyze the execution output, determine if there were any bugs, and provide a summary of the findings. "
            ),
            "Research idea": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        with capture_llm_calls() as refs:
            response = cast(
                dict,
                query(
                    system_message=prompt,
                    user_message=None,
                    func_spec=review_func_spec,
                    model=self.cfg.agent.feedback.model,
                    temperature=self.cfg.agent.feedback.temp,
                ),
            )

        node.analysis = response["summary"]
        node.agent_review_bug_advisory = response["is_bug"]
        node.llm_call_refs.extend(refs)
        # GLM review is explanatory only. It cannot hide a successful host
        # execution (or rescue an execution exception) from deterministic gates.
        node.is_buggy = node.exc_type is not None
        print(
            "[red]Checking if response contains metric name and description[/red]",
            flush=True,
        )
        print("[red]Execution review received[/red]")

    def _generate_plotting_code(
        self, node: Node, working_dir: str, plot_code_from_prev_stage: str = None
    ) -> str:
        """Generate code for plotting experiment results"""
        prompt_guideline = [
            "AVAILABLE DATA: ",
            "Experiment Data: experiment_data.npy",
        ]
        prompt_guideline += [
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "Include training/validation curves if available",
            "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
            "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Use the following experiment code to infer the data to plot: " + node.code,
            "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
        ]
        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data = np.load(os.path.join(working_dir, 'experiment_data.npy'), allow_pickle=True).item()
                except Exception as e:
                    print(f'Error loading experiment data: {{e}}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {{e}}")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {{e}}")
                    plt.close()
            """,
        ]
        # add instruction for format
        plotting_prompt = {
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= self._prompt_resp_fmt
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }

        # For stage 3, initialize with stage 2's plotting code
        if (
            self.stage_name
            and self.stage_name.startswith("3_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: Use the following base plotting code as a starting point:",
                    "Base plotting code: " + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between different datasets",
                    "3. Add dataset-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from which dataset",
                    "5. Use consistent naming conventions for saved files",
                ]
            )
        # For stage 4, initialize with stage 3's plotting code
        elif (
            self.stage_name
            and self.stage_name.startswith("4_")
            and plot_code_from_prev_stage
        ):
            prompt_guideline.extend(
                [
                    "IMPORTANT: This is an ablation study. Use the following base plotting code as a starting point:",
                    "Base plotting code: \n" + plot_code_from_prev_stage,
                    "Modify the base plotting code to:",
                    "1. Keep the same numpy data structure and plotting style",
                    "2. Add comparison plots between ablation and baseline results",
                    "3. Add ablation-specific visualizations if needed",
                    "4. Include clear labels indicating which plots are from ablation vs baseline",
                    "5. Use consistent naming conventions for saved files",
                ]
            )

        # Get plotting code from LLM
        plan, code = self.plan_and_code_query(plotting_prompt)

        # Ensure the code starts with imports
        if not code.strip().startswith("import"):
            code = "import matplotlib.pyplot as plt\nimport numpy as np\n\n" + code

        node.plot_code = code
        node.plot_plan = plan

        return code

    def _determine_datasets_successfully_tested(self, node: Node) -> List[str]:
        """Determine which datasets are successfully tested based on VLM feedback"""
        plot_analyses = ""
        for i, plot_analysis in enumerate(node.plot_analyses):
            plot_analyses += f"plot {i+1}: {plot_analysis['analysis']}\n"

        determine_prompt = {
            "Introduction": "You are an AI researcher analyzing experiment results. Based on the plot analyses and feedback, determine which datasets are successfully tested. Return reasoning and the dataset names that are successfully executed, or an empty string if no datasets are successfully executed.",
            "Plot analyses": plot_analyses,
            "VLM feedback summary": node.vlm_feedback_summary,
            "Original plotting code": node.plot_code,
            "Response format": (
                "Your response should start with 'REASONING: <reasoning>' to think about the plot analysis and feedback in the first line."
                "In the second line, you should have a list of dataset names that are successfully executed, starting with 'SUCCESSFULLY_TESTED_DATASETS: <list_datasets_successfully_tested>', "
            ),
        }

        retry_count = 0
        retry_limit = MAX_RESEARCH_DECISION_RETRIES
        while retry_count < retry_limit:
            response = query(
                system_message=determine_prompt,
                user_message=None,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            )

            (
                reasoning,
                datasets_successfully_tested_str,
            ) = _parse_keyword_prefix_response(
                response, "REASONING:", "SUCCESSFULLY_TESTED_DATASETS:"
            )
            if reasoning is not None and datasets_successfully_tested_str is not None:
                if datasets_successfully_tested_str == "":
                    return [""]
                # Split by comma and clean each dataset name
                datasets = [
                    ds.strip() for ds in datasets_successfully_tested_str.split(",")
                ]
                # Filter out empty strings and ensure all elements are strings
                datasets = [ds for ds in datasets if isinstance(ds, str) and ds]
                logger.info(f"Successfully parsed datasets: {datasets}")
                return datasets

            retry_count += 1
            logger.warning(
                f"Failed to parse successfully tested datasets response (attempt {retry_count}/{retry_limit})"
            )

        logger.error(
            f"Failed to parse successfully tested datasets response after {retry_limit} retries. Falling back to an empty list."
        )
        return [""]

    def _analyze_plots_with_vlm(self, node: Node) -> None:
        """Analyze experimental plots using VLM"""
        if not node.plot_paths:
            return

        # for debugging
        print(f"[cyan]Plots available for review: {len(node.plot_paths)}[/cyan]")

        def encode_image_to_base64(image_path):
            with open(image_path, "rb") as image_file:
                try:
                    return base64.b64encode(image_file.read()).decode("utf-8")
                except Exception as e:
                    print(f"[red]Error encoding image: {type(e).__name__}[/red]")
                    return None

        if not len(node.plot_paths) > 10:
            selected_plots = node.plot_paths
        else:
            print(
                f"[red]Warning: {len(node.plot_paths)} plots received, this may be too many to analyze effectively. Calling LLM to select the most relevant plots to analyze.[/red]"
            )
            # select 10 plots to analyze
            prompt_select_plots = {
                "Introduction": (
                    "You are an experienced AI researcher analyzing experimental results. "
                    "You have been provided with plots from a machine learning experiment. "
                    "Please select 10 most relevant plots to analyze. "
                    "For similar plots (e.g. generated samples at each epoch), select only at most 5 plots at a suitable interval of epochs."
                    "Format your response as a list of plot paths, where each plot path includes the full path to the plot file."
                ),
                "Plot paths": node.plot_paths,
            }

            try:
                response_select_plots = cast(
                    dict,
                    query(
                        system_message=prompt_select_plots,
                        user_message=None,
                        func_spec=plot_selection_spec,
                        model=self.cfg.agent.feedback.model,
                        temperature=self.cfg.agent.feedback.temp,
                    ),
                )

                print("[cyan]Plot selection response received[/cyan]")
                # Extract the plot paths list
                selected_plots = response_select_plots.get("selected_plots", [])

                # Validate that all paths exist and are image files
                allowed_plots = set(node.plot_paths)
                valid_plots = []
                for plot_path in selected_plots:
                    if (
                        isinstance(plot_path, str)
                        and plot_path in allowed_plots
                        and os.path.exists(plot_path)
                        and plot_path.lower().endswith((".png", ".jpg", ".jpeg"))
                    ):
                        valid_plots.append(plot_path)
                    else:
                        logger.warning("Research agent selected an invalid plot path")

                # Use the validated list
                if valid_plots:
                    print(f"[cyan]Selected valid plots: {len(valid_plots)}[/cyan]")
                    selected_plots = valid_plots
                else:
                    raise FunctionCallValidationError(
                        "Plot selection did not reference an allowed plot"
                    )

            except Exception as e:
                if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                    raise
                logger.error(
                    "Plot-selection research decision failed: %s", type(e).__name__
                )
                raise ResearchDecisionError(
                    "Plot-selection research decision failed"
                ) from None

        print("[cyan]Before encoding images[/cyan]")
        user_message = [
            {
                "type": "text",
                "text": (
                    "You are an experienced AI researcher analyzing experimental results. "
                    "You have been provided with plots from a machine learning experiment. "
                    f"This experiment is based on the following research idea: {self.task_desc}"
                    "Please analyze these plots and provide detailed insights about the results. "
                    "If you don't receive any plots, say 'No plots received'. "
                    "Never make up plot analysis. "
                    "Please return the analyzes with strict order of uploaded images, but DO NOT include any word "
                    "like 'the first plot'."
                ),
            }
        ] + [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image_to_base64(plot_path)}"
                },
            }
            for plot_path in selected_plots
        ]

        response = cast(
            dict,
            query(
                system_message=None,
                user_message=user_message,
                func_spec=vlm_feedback_spec,
                model=self.cfg.agent.vlm_feedback.model,
                temperature=self.cfg.agent.vlm_feedback.temp,
            ),
        )
        print(
            f"[cyan]VLM response received from {self.cfg.agent.vlm_feedback.model}[/cyan]"
        )
        analyses = response["plot_analyses"]
        if len(analyses) != len(selected_plots):
            raise FunctionCallValidationError(
                "Plot analyses do not match the selected evidence set"
            )
        if response["valid_plots_received"]:
            node.is_buggy_plots = False
        else:
            node.is_buggy_plots = True

        for analysis, plot_path in zip(analyses, selected_plots):
            analysis["plot_path"] = plot_path

        node.plot_analyses = analyses
        node.vlm_feedback_summary = response["vlm_feedback_summary"]

        # Prefer dataset names extracted from parsed metrics (deterministic).
        # Fall back to inferring from plot analyses only when missing.
        if (
            not node.datasets_successfully_tested
            or node.datasets_successfully_tested == [""]
        ):
            node.datasets_successfully_tested = (
                self._determine_datasets_successfully_tested(node)
            )

    def _generate_node_summary(self, node: Node) -> dict:
        """Generate a summary of the node's experimental findings"""
        summary_prompt = {
            "Introduction": (
                "You are an AI researcher analyzing experimental results. "
                "Please summarize the findings from this experiment iteration."
            ),
            "Research idea": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Plan": node.plan,
            "Execution output": wrap_code(node.term_out, lang=""),
            "Analysis": node.analysis,
            "Metric": str(node.metric) if node.metric else "Failed",
            "Plot Analyses": (
                node.plot_analyses if hasattr(node, "plot_analyses") else []
            ),
            "VLM Feedback": (
                node.vlm_feedback_summary
                if hasattr(node, "vlm_feedback_summary")
                else ""
            ),
        }

        return cast(
            dict,
            query(
                system_message=summary_prompt,
                user_message=None,
                func_spec=experiment_summary_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
            ),
        )


class GPUManager:
    """Manages GPU allocation across processes"""

    def __init__(self, devices: Sequence[str]):
        self.devices = tuple(str(device) for device in devices)
        self.num_gpus = len(self.devices)
        self.available_gpus: list[str] = list(self.devices)
        self.gpu_assignments: Dict[str, str] = {}

    def acquire_gpu(self, process_id: str) -> str:
        """Assigns a GPU to a process"""
        if not self.available_gpus:
            raise RuntimeError("No GPUs available")
        gpu_id = self.available_gpus.pop(0)
        self.gpu_assignments[process_id] = gpu_id
        return gpu_id

    def release_gpu(self, process_id: str):
        """Releases GPU assigned to a process"""
        if process_id in self.gpu_assignments:
            gpu_id = self.gpu_assignments[process_id]
            self.available_gpus.append(gpu_id)
            del self.gpu_assignments[process_id]


def get_gpu_devices() -> list[str]:
    """Return the exact CUDA device tokens authorized by the parent process."""

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        devices = [token.strip() for token in raw.split(",") if token.strip()]
        if not devices or devices == ["-1"]:
            return []
        if "-1" in devices or len(devices) != len(set(devices)):
            return []
        return devices
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [line.strip() for line in nvidia_smi.stdout.splitlines() if line.strip()]
    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def get_gpu_count() -> int:
    """Compatibility wrapper for callers that need only the authorized count."""

    return len(get_gpu_devices())


class ParallelAgent:
    def __init__(
        self,
        task_desc: str,
        cfg: Config,
        journal: Journal,
        stage_name=None,
        best_stage3_node=None,
        best_stage2_node=None,
        best_stage1_node=None,
        evaluation_metric: str | None = None,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.journal = journal
        self.stage_name = stage_name
        self.best_stage3_node = (
            best_stage3_node  # to initialize ablation stuides (stage 4)
        )
        self.best_stage1_node = (
            best_stage1_node  # to initialize hyperparam tuning (stage 2)
        )
        self.best_stage2_node = (
            best_stage2_node  # to initialize plotting code (stage 3)
        )
        self.data_preview = None
        configured_workers = cfg.agent.num_workers
        if (
            isinstance(configured_workers, bool)
            or not isinstance(configured_workers, int)
            or not 1 <= configured_workers <= MAX_PARALLEL_WORKERS
        ):
            raise ValueError("agent.num_workers must be an integer between 1 and 64")
        self.num_workers = configured_workers
        self.gpu_devices = get_gpu_devices()
        self.num_gpus = len(self.gpu_devices)
        print(f"num_gpus: {self.num_gpus}")
        if self.num_gpus == 0:
            print("No GPUs detected, falling back to CPU-only mode")
        else:
            print(f"Detected {self.num_gpus} GPUs")

        self.gpu_manager = GPUManager(self.gpu_devices) if self.num_gpus > 0 else None

        if self.num_gpus > 0:
            self.num_workers = min(self.num_workers, self.num_gpus)
            logger.info(f"Limiting workers to {self.num_workers} to match GPU count")

        self.timeout = self.cfg.exec.timeout
        self._is_shutdown = False
        # The manager persists this research decision in the checkpoint. A
        # resumed pending-seed stage must never ask the model to redefine it.
        self.evaluation_metrics = (
            evaluation_metric
            if isinstance(evaluation_metric, str) and evaluation_metric.strip()
            else self._define_global_metrics()
        )
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
        self._ablation_state = {  # store ablation names
            "completed_ablations": {
                key
                for candidate in self.journal.nodes
                if (
                    key := _ablation_idea_key(
                        candidate.ablation_name,
                        candidate.ablation_component,
                    )
                )
            },
        }
        self._hyperparam_tuning_state = {  # store hyperparam tuning ideas
            "tried_hyperparams": {
                key
                for candidate in self.journal.nodes
                if (key := _canonical_idea_key(candidate.hyperparam_name))
            },
        }

    def _define_global_metrics(self) -> str:
        """Define eval metric to be used across all experiments"""
        prompt = {
            "Introduction": (
                "You are an AI researcher setting up experiments. "
                "Please propose meaningful evaluation metrics that will help analyze "
                "the performance and characteristics of solutions for this research task."
            ),
            "Research idea": self.task_desc,
            "Instructions": [
                "Select exactly one metric from the host-supported enum in the function schema.",
                "Choose the metric that best tests the stated hypothesis.",
                "Validation loss is tracked separately.",
            ],
        }

        response = query(
            system_message=prompt,
            user_message=None,
            func_spec=metric_selection_spec,
            model=self.cfg.agent.code.model,
            temperature=self.cfg.agent.code.temp,
        )
        metric = response["metric"]
        if metric not in SUPPORTED_DETERMINISTIC_METRICS:
            raise FunctionCallValidationError(
                "Research Agent selected an unsupported deterministic metric"
            )
        print("[green]Research Agent selected the stage metric contract[/green]")
        return metric

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        retries = _validate_retry_count(retries)
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )

            try:
                return extract_single_plan_and_code(completion_text)
            except ValueError:
                pass
            print("Plan + code extraction failed, retrying...")
            prompt["Parsing Feedback"] = (
                "The code extraction failed. Make sure to use the format ```python ... ``` for the code blocks."
            )
        raise ResearchDecisionError(
            "Research Agent returned malformed plan/code after bounded retries"
        )

    def _generate_seed_eval_aggregation_node(
        self, node: Node, agg_plotting_code: str
    ) -> Node:
        """Generate a special aggregation node for seed evaluation results"""
        return Node(
            plan="Aggregate results from multiple seeds",
            code="# plotting aggregation code",
            plot_code=agg_plotting_code,
            parent=node,
            is_seed_node=True,
            is_seed_agg_node=True,
        )

    def _run_multi_seed_evaluation(self, node: Node) -> List[Node]:
        """Run multiple seeds of the same node to get statistical metrics.
        Returns a list of nodes with different random seeds."""

        # Convert node to dict for parallel processing
        node_data = node.to_dict()
        node_code = node.code

        seed_cfg = self.cfg.agent.multi_seed_eval
        seeds_to_run = _configured_multi_seed_values(seed_cfg)
        _validate_confirmation_seed_set(node_code, seeds_to_run)

        staged_results: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        expected_contract = node.evaluation_comparison_contract
        expected_signature = node.metric.comparison_signature
        if expected_contract is None:
            raise ExperimentCannotContinueError(
                "Qualified node lacks a comparison-ready evaluation"
            )

        # A wave never contains more work than the executor (or GPU pool) can
        # actually run. Each wave has its own execution timeout and releases
        # its GPU leases before the next wave is submitted. Results remain in
        # memory until every wave has passed validation.
        try:
            for offset in range(0, len(seeds_to_run), self.num_workers):
                wave_seeds = seeds_to_run[offset : offset + self.num_workers]
                futures = []
                future_metadata: Dict[Any, tuple[str, int, str, str]] = {}
                wave_process_ids: list[str] = []
                try:
                    for seed in wave_seeds:
                        gpu_id = None
                        process_id = f"seed_{seed}_worker"
                        if self.gpu_manager is not None:
                            gpu_id = self.gpu_manager.acquire_gpu(process_id)
                            wave_process_ids.append(process_id)
                            logger.info("Assigned a GPU to multi-seed worker")

                        seeded_code, bootstrap_hash = _inject_seed_bootstrap(
                            node_code, seed
                        )
                        seed_node_data = copy.deepcopy(node_data)
                        seed_node_data["code"] = seeded_code
                        seed_node_data["random_seed"] = seed
                        seed_node_data["seed_bootstrap_hash"] = bootstrap_hash

                        print("[yellow]Starting multi-seed eval...[/yellow]")
                        future = self.executor.submit(
                            self._process_node_wrapper,
                            seed_node_data,
                            self.task_desc,
                            self.cfg,
                            gpu_id,
                            "",
                            self.evaluation_metrics,
                            self.stage_name,
                            None,
                            None,
                            None,
                            None,
                            None,
                            True,
                        )
                        futures.append(future)
                        future_metadata[future] = (
                            process_id,
                            seed,
                            bootstrap_hash,
                            seeded_code,
                        )

                    _done, unfinished = wait(
                        futures,
                        timeout=self.timeout,
                        return_when=ALL_COMPLETED,
                    )
                    if unfinished:
                        raise TimeoutError("multi-seed wave deadline exceeded")
                    for future in futures:
                        (
                            _process_id,
                            seed,
                            bootstrap_hash,
                            seeded_code,
                        ) = future_metadata[future]
                        result_data = future.result()
                        if not isinstance(result_data, dict):
                            raise ExperimentCannotContinueError(
                                "Multi-seed worker returned an invalid result"
                            )
                        result_node = Node.from_dict(copy.deepcopy(result_data))
                        if (
                            result_data.get("parent_id") != node.id
                            or result_node.random_seed != seed
                            or result_node.seed_bootstrap_hash != bootstrap_hash
                            or result_node.code != seeded_code
                            or result_node.is_seed_node is not True
                            or result_node.is_seed_agg_node is True
                            or not result_node.has_verified_metric
                            or result_node.evaluation_comparison_contract
                            != expected_contract
                            or result_node.metric.comparison_signature
                            != expected_signature
                            or result_node.id in seen_node_ids
                            or self.journal.get_node_by_id(result_node.id) is not None
                        ):
                            raise ExperimentCannotContinueError(
                                "Multi-seed result does not match the qualified evidence contract"
                            )
                        seen_node_ids.add(result_node.id)
                        staged_results.append(copy.deepcopy(result_data))
                finally:
                    if self.gpu_manager is not None:
                        for process_id in wave_process_ids:
                            self.gpu_manager.release_gpu(process_id)
        except BaseException as exc:
            for pending in locals().get("futures", []):
                pending.cancel()
            self.cleanup()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
                raise
            logger.error("Multi-seed evaluation failed: %s", type(exc).__name__)
            raise ExperimentCannotContinueError(
                "Multi-seed evaluation did not complete every required seed"
            ) from None

        journal_size = len(self.journal.nodes)
        parent = self.journal.get_node_by_id(node.id)
        if parent is None:
            raise ExperimentCannotContinueError(
                "Qualified node is missing from the stage journal"
            )
        original_children = set(parent.children)
        seed_nodes: list[Node] = []
        try:
            for result_data in staged_results:
                result_node = Node.from_dict(copy.deepcopy(result_data), self.journal)
                self.journal.append(result_node)
                seed_nodes.append(result_node)
        except BaseException as exc:
            del self.journal.nodes[journal_size:]
            parent.children = original_children
            if not isinstance(exc, Exception):
                raise
            raise ExperimentCannotContinueError(
                "Multi-seed evidence could not be committed atomically"
            ) from None
        return seed_nodes

    def _run_plot_aggregation(self, node: Node, seed_nodes: List[Node]) -> Node:
        """Generate an aggregation node for seed evaluation results"""
        if len(seed_nodes) >= MIN_SCIENTIFIC_SEEDS and all(
            seed.has_verified_metric for seed in seed_nodes
        ):
            try:
                from .interpreter import Interpreter

                # Create aggregation plotting code
                agg_plotting_code = self._aggregate_seed_eval_results(seed_nodes, node)

                # Create a special aggregation node
                agg_node = self._generate_seed_eval_aggregation_node(
                    node, agg_plotting_code
                )
                agg_node.parent = node

                # Execute aggregation plotting code
                print("[blue]Creating Interpreter for seed node aggregation[/blue]")
                process_interpreter = _interpreter_for_workspace(
                    self.cfg,
                    self.cfg.workspace_dir,
                    allow_network=False,
                )

                try:
                    working_dir = process_interpreter.working_dir
                    plot_exec_result = process_interpreter.run(agg_plotting_code, True)
                    logger.info(
                        "Aggregation plotting finished (exception_type=%s)",
                        plot_exec_result.exc_type,
                    )
                    process_interpreter.cleanup_session()
                    # Save aggregated plots
                    plots_dir = Path(working_dir) / "working"
                    if plots_dir.exists():
                        exp_results_dir = (
                            Path(self.cfg.log_dir).resolve()
                            / "experiment_results"
                            / f"seed_aggregation_{agg_node.id}"
                        )
                        _publish_source_artifacts(
                            exp_results_dir,
                            {"aggregation_plotting_code.py": agg_plotting_code},
                        )

                        # Copy only bounded regular PNGs. Agent-created symlinks
                        # are rejected and source files are never moved by the host.
                        for final_path in _publish_plot_artifacts(
                            plots_dir,
                            exp_results_dir,
                        ):
                            web_path = f"../../logs/{Path(self.cfg.workspace_dir).name}/experiment_results/seed_aggregation_{agg_node.id}/{final_path.name}"
                            agg_node.plots.append(web_path)
                            agg_node.plot_paths.append(str(final_path.absolute()))

                    agg_node.is_buggy = False
                    agg_node.exp_results_dir = exp_results_dir
                    agg_node_dict = agg_node.to_dict()
                    agg_node_new = Node.from_dict(
                        agg_node_dict, self.journal
                    )  # to update the parent-child relationship in the journal
                    # Add aggregation node to journal
                    self.journal.append(agg_node_new)
                    return agg_node_new
                finally:
                    if process_interpreter:
                        process_interpreter.cleanup_session()

            except Exception as e:
                if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                    raise
                logger.error("Seed result aggregation failed: %s", type(e).__name__)
                raise ExperimentCannotContinueError(
                    "Seed result aggregation failed"
                ) from None
        else:
            raise ExperimentCannotContinueError(
                "Seed aggregation requires at least three verified seed results"
            )

    @staticmethod
    def _process_node_wrapper(
        node_data,
        task_desc,
        cfg,
        gpu_id: str | None = None,
        memory_summary: str = None,
        evaluation_metrics=None,
        stage_name=None,
        new_ablation_idea=None,
        new_hyperparam_idea=None,
        best_stage3_plot_code=None,
        best_stage2_plot_code=None,
        best_stage1_plot_code=None,
        seed_eval=False,
        context_pack_ref=None,
    ):
        """Wrapper function that creates a fresh environment for each process"""
        from .journal import Node, Journal
        from copy import deepcopy
        import os
        import multiprocessing

        print("Starting _process_node_wrapper")

        # Create process-specific workspace
        process_id = multiprocessing.current_process().name
        workspace = os.path.join(cfg.workspace_dir, f"process_{process_id}")
        os.makedirs(workspace, exist_ok=True)
        print(f"Process {process_id} using workspace: {workspace}")
        # Create process-specific working directory
        working_dir = os.path.join(workspace, "working")
        os.makedirs(working_dir, exist_ok=True)

        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info(f"Process {process_id} assigned to GPU {gpu_id}")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info(f"Process {process_id} running on CPU")

        # Create minimal agent for worker process with the global metric definition
        worker_agent = MinimalAgent(
            task_desc=task_desc,
            cfg=cfg,
            memory_summary=memory_summary,
            evaluation_metrics=evaluation_metrics,
            stage_name=stage_name,
        )

        # Create interpreter instance for worker process
        print("Creating Interpreter")
        experiment_interpreter = _interpreter_for_workspace(
            cfg,
            workspace,
            allow_network=_experiment_network_enabled(cfg),
        )
        analysis_interpreter = (
            None
            if seed_eval
            else _interpreter_for_workspace(
                cfg,
                workspace,
                allow_network=False,
            )
        )

        try:
            print(f"stage_name: {stage_name}")
            # Recreate node object from node_data, which becomes a parent node.
            if node_data:
                parent_node = Node.from_dict(node_data, journal=None)
                print(f"Recreated parent node: {parent_node.id}")
            else:
                parent_node = None
                print("No parent node to recreate")

            # Process the node using worker agent
            print("Starting node processing")
            if seed_eval:
                # Use the parent node's code to run the same code again
                child_node = worker_agent._generate_seed_node(parent_node)
                child_node.parent = parent_node
                # Plot code should also be the same as the parent node
                child_node.plot_code = parent_node.plot_code
            else:
                if parent_node is None:
                    print("Drafting new node")
                    child_node = worker_agent._draft()
                elif parent_node.is_buggy:
                    print("Debugging node with id: ", parent_node.id)
                    child_node = worker_agent._debug(parent_node)
                    child_node.parent = parent_node
                else:
                    if (
                        new_hyperparam_idea is not None and new_ablation_idea is None
                    ):  # stage 2
                        child_node = worker_agent._generate_hyperparam_tuning_node(
                            parent_node, new_hyperparam_idea
                        )
                        child_node.parent = parent_node
                        logger.info(
                            f"Processing hyperparam tuning: {child_node.hyperparam_name}"
                        )
                        print(
                            f"[cyan]Running hyperparam tuning: {child_node.hyperparam_name}[/cyan]"
                        )
                    elif (
                        new_ablation_idea is not None and new_hyperparam_idea is None
                    ):  # stage 4
                        child_node = worker_agent._generate_ablation_node(
                            parent_node, new_ablation_idea
                        )
                        child_node.parent = parent_node
                        logger.info(f"Processing ablation: {child_node.ablation_name}")
                        print(
                            f"[cyan]Running ablation study: {child_node.ablation_name}[/cyan]"
                        )
                    else:
                        print("Improving node with id: ", parent_node.id)
                        child_node = worker_agent._improve(parent_node)
                        child_node.parent = parent_node

            if context_pack_ref:
                child_node.context_pack_refs = [str(context_pack_ref)]

            # Execute and parse results
            print("Running code")
            experiment_data_path = Path(working_dir) / "experiment_data.npy"
            try:
                experiment_data_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not remove stale experiment data: %s",
                    type(exc).__name__,
                )
            exec_result = experiment_interpreter.run(child_node.code, True)
            experiment_interpreter.cleanup_session()

            print("Parsing execution results")
            if seed_eval:
                # Confirmation seeds are a host-only replay path. Advisory LLM
                # review, plotting, and VLM cannot veto or consume the budget
                # of a deterministic multi-seed transaction.
                child_node.absorb_exec_result(exec_result)
                child_node.analysis = "Host-only confirmation-seed replay"
                child_node.agent_review_bug_advisory = None
                child_node.is_buggy = child_node.exc_type is not None
            else:
                worker_agent.parse_exec_result(
                    node=child_node,
                    exec_result=exec_result,
                    workspace=working_dir,
                )

            # Add check for saved data files
            data_files = (
                ["experiment_data.npy"] if experiment_data_path.is_file() else []
            )
            if not data_files:
                logger.warning(
                    "No .npy files found in working directory. Data may not have been saved properly."
                )
            evaluation_report = evaluate_experiment_data(
                experiment_data_path,
                requested_metric=worker_agent.evaluation_metrics,
            )
            child_node.evaluation_report = evaluation_report
            deterministic_metric = evaluation_report.get("metric")
            if evaluation_report.get("status") == "verified" and isinstance(
                deterministic_metric, dict
            ):
                evidence_dir = (
                    Path(cfg.log_dir).resolve()
                    / "experiment_results"
                    / f"experiment_{child_node.id}"
                )
                _preserve_evaluation_artifact(
                    experiment_data_path,
                    evidence_dir / "experiment_data.npy",
                    expected_hash=evaluation_report["input"]["sha256"],
                )
                _publish_source_artifacts(
                    evidence_dir,
                    {"experiment_code.py": child_node.code},
                )
                child_node.exp_results_dir = str(evidence_dir)
                child_node.metric = MetricValue(value=deterministic_metric)
                child_node.metric_provenance = "deterministic_verified"
                child_node.advisory_metric = None
                child_node.datasets_successfully_tested = (
                    _extract_dataset_names_from_metric(child_node.metric)
                )
                logger.info(
                    "Deterministically verified %s for node %s from experiment_data.npy",
                    evaluation_report.get("selected_metric"),
                    child_node.id,
                )
            elif not data_files:
                child_node.metric = WorstMetricValue()
                child_node.metric_provenance = "unavailable"
                child_node.is_buggy = True
                child_node.datasets_successfully_tested = []

            if (
                data_files
                and not seed_eval
                and evaluation_report.get("safe_for_legacy_parser") is True
                and child_node.metric_provenance != "deterministic_verified"
            ):
                # Call LLM to parse data files and extract advisory metrics.
                parse_metrics_prompt = {
                    "Introduction": (
                        "You are an AI researcher analyzing experimental results stored in numpy files. "
                        "Write code to load and analyze the metrics from experiment_data.npy."
                    ),
                    "Primary Evaluation Metric (optimize this)": worker_agent.evaluation_metrics,
                    "Context": [
                        "Original Code: " + child_node.code,
                    ],
                    "Instructions": [
                        "0. Make sure to get the working directory from os.path.join(os.getcwd(), 'working')",
                        "1. Load the experiment_data.npy file, which is located in the working directory",
                        "2. Extract ONLY the primary evaluation metric above for each dataset (prefer validation split if present).",
                        "3. Always print the name of the dataset before printing the metrics.",
                        "4. Always print the name of the metric before printing the value by specifying the metric name clearly (e.g. 'validation accuracy').",
                        "5. You only need to print the best or final value for this metric for each dataset.",
                        "6. Do NOT print losses or additional metrics; keep output minimal and focused on the primary metric.",
                        "7. DO NOT CREATE ANY PLOTS",
                        "Important code structure requirements:",
                        "  - Do NOT put any execution code inside 'if __name__ == \"__main__\":' block. Do not use 'if __name__ == \"__main__\":' at all.",
                        "  - All code should be at the global scope or in functions that are called from the global scope",
                        "  - The script should execute immediately when run, without requiring any special entry point",
                    ],
                    "Example data loading code": [
                        """
                            import matplotlib.pyplot as plt
                            import numpy as np

                            experiment_data = np.load(os.path.join(os.getcwd(), 'experiment_data.npy'), allow_pickle=True).item()
                            """
                    ],
                    "Response format": worker_agent._prompt_metricparse_resp_fmt(),
                }

                (
                    parse_metrics_plan,
                    parse_metrics_code,
                ) = worker_agent.plan_and_code_query(parse_metrics_prompt)
                logger.info(
                    "Metric parser generated (plan_ref=%s code_ref=%s)",
                    _opaque_content_ref(parse_metrics_plan),
                    _opaque_content_ref(parse_metrics_code),
                )
                child_node.parse_metrics_plan = parse_metrics_plan
                child_node.parse_metrics_code = parse_metrics_code
                try:
                    # Execute the parsing code
                    metrics_exec_result = analysis_interpreter.run(
                        parse_metrics_code, True
                    )
                    analysis_interpreter.cleanup_session()
                    child_node.parse_term_out = metrics_exec_result.term_out
                    child_node.parse_exc_type = metrics_exec_result.exc_type
                    child_node.parse_exc_info = metrics_exec_result.exc_info
                    child_node.parse_exc_stack = metrics_exec_result.exc_stack

                    if metrics_exec_result.exc_type is None:
                        # Extract metrics from the execution output
                        metrics_prompt = {
                            "Introduction": (
                                "Parse the primary evaluation metric from the execution output. "
                                "Return only ONE metric (the primary metric) with final/best values for each dataset."
                            ),
                            "Primary Evaluation Metric (optimize this)": worker_agent.evaluation_metrics,
                            "Execution Output": metrics_exec_result.term_out,
                        }
                        print("[blue]Metrics parsing execution completed[/blue]")

                        metrics_response = cast(
                            dict,
                            query(
                                system_message=metrics_prompt,
                                user_message=None,
                                func_spec=metric_parse_spec,
                                model=cfg.agent.feedback.model,
                                temperature=cfg.agent.feedback.temp,
                            ),
                        )
                        # If there is any None value, child_node.metric should be set to WorstMetricValue.
                        # This is achieved by raising an error in the MetricValue class,
                        # which sets child_node.is_buggy to True, thereby
                        # causing child_node.metric to be assigned WorstMetricValue.
                        print("[blue]Structured metrics response received[/blue]")
                        if metrics_response["valid_metrics_received"]:
                            advisory_metric = MetricValue(
                                value={"metric_names": metrics_response["metric_names"]}
                            )
                            child_node.advisory_metric = advisory_metric.value
                            child_node.metric = WorstMetricValue()
                            child_node.metric_provenance = "agent_reported_advisory"
                            child_node.datasets_successfully_tested = []
                            logger.info(
                                "Stored an advisory agent-reported metric for node %s; "
                                "it is excluded from ranking and stage gates",
                                child_node.id,
                            )
                        else:
                            child_node.metric = WorstMetricValue()
                            child_node.metric_provenance = "unavailable"
                            child_node.is_buggy = True
                            logger.error(
                                f"No valid metrics received for node {child_node.id}"
                            )
                            child_node.datasets_successfully_tested = []
                    else:
                        logger.error(
                            "Metric parser execution failed: %s",
                            metrics_exec_result.exc_type or "unknown",
                        )
                        child_node.metric = WorstMetricValue()
                        child_node.metric_provenance = "unavailable"
                        child_node.is_buggy = True
                        child_node.datasets_successfully_tested = []

                except Exception as e:
                    if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(
                        e
                    ):
                        raise
                    logger.error(
                        "Error parsing metrics for node %s: %s",
                        child_node.id,
                        type(e).__name__,
                    )
                    child_node.metric = WorstMetricValue()
                    child_node.metric_provenance = "unavailable"
                    child_node.is_buggy = True
                    child_node.parse_exc_type = type(e).__name__
                    child_node.parse_exc_info = None
                    child_node.parse_exc_stack = None
                    child_node.parse_term_out = (
                        "Metric parsing failed with " + type(e).__name__
                    )
                    child_node.datasets_successfully_tested = []
            elif (
                data_files and child_node.metric_provenance != "deterministic_verified"
            ):
                child_node.metric = WorstMetricValue()
                child_node.metric_provenance = "unavailable"
                child_node.is_buggy = True
                child_node.datasets_successfully_tested = []
                logger.error(
                    "Refusing legacy metric parsing for unsafe experiment_data.npy: %s",
                    evaluation_report.get("reason"),
                )

            # if experiment was successful, generate and run plotting code
            if not child_node.is_buggy and not seed_eval:
                try:
                    retry_count = 0
                    while True:
                        if seed_eval:
                            # Use the parent node's plotting code instead of generating new one
                            plotting_code = parent_node.plot_code
                        else:
                            if (
                                worker_agent.stage_name
                                and worker_agent.stage_name.startswith("3_")
                                and best_stage2_plot_code
                            ):
                                plot_code_from_prev_stage = best_stage2_plot_code
                            elif (
                                worker_agent.stage_name
                                and worker_agent.stage_name.startswith("4_")
                                and best_stage3_plot_code
                            ):
                                plot_code_from_prev_stage = best_stage3_plot_code
                            else:
                                plot_code_from_prev_stage = None

                            plotting_code = worker_agent._generate_plotting_code(
                                child_node, working_dir, plot_code_from_prev_stage
                            )
                        plot_exec_result = analysis_interpreter.run(plotting_code, True)
                        analysis_interpreter.cleanup_session()
                        child_node.absorb_plot_exec_result(plot_exec_result)
                        if child_node.plot_exc_type and retry_count < 3:
                            print(
                                f"[red]Plotting code failed with exception: {child_node.plot_exc_type}[/red]"
                            )
                            print(
                                "[red]Plotting failed; code and output remain in the node artifact[/red]"
                            )
                            retry_count += 1
                            continue
                        else:
                            break

                    logger.info(
                        "Plotting finished (exception_type=%s)",
                        plot_exec_result.exc_type,
                    )
                    # Track generated plots
                    plots_dir = Path(working_dir)
                    if plots_dir.exists():
                        print("Plots directory exists, saving plots to node")
                        # Save the plotting code first
                        exp_results_dir = Path(child_node.exp_results_dir)
                        child_node.exp_results_dir = exp_results_dir
                        _publish_source_artifacts(
                            exp_results_dir,
                            {
                                "plotting_code.py": plotting_code,
                                "experiment_code.py": child_node.code,
                            },
                        )
                        plot_code_path = exp_results_dir / "plotting_code.py"
                        logger.info(f"Saved plotting code to {plot_code_path}")
                        exp_code_path = exp_results_dir / "experiment_code.py"
                        logger.info(f"Saved experiment code to {exp_code_path}")
                        # Never publish plotting-generated arrays into the
                        # canonical evidence directory. In particular,
                        # experiment_data.npy is immutable after evaluation.
                        for final_path in _publish_plot_artifacts(
                            plots_dir,
                            exp_results_dir,
                        ):
                            # Create a web-friendly relative path starting from logs directory
                            web_path = f"../../logs/{Path(cfg.workspace_dir).name}/experiment_results/experiment_{child_node.id}/{final_path.name}"

                            child_node.plots.append(web_path)  # For visualization
                            child_node.plot_paths.append(
                                str(final_path.absolute())
                            )  # For programmatic access

                            logger.info(
                                f"[green]Generated plot: {final_path.stem}[/green]"
                            )
                            logger.debug(f"Plot absolute path: {final_path.absolute()}")
                            logger.debug(f"Plot web path: {web_path}")
                except Exception as e:
                    if is_llm_budget_exception(e):
                        raise
                    logger.error(
                        "Error generating plots for node %s: %s",
                        child_node.id,
                        type(e).__name__,
                    )

                if child_node.plots:
                    try:
                        worker_agent._analyze_plots_with_vlm(child_node)
                        logger.info(
                            f"Generated VLM analysis for plots in node {child_node.id}"
                        )
                    except Exception as e:
                        if isinstance(
                            e, ResearchDecisionError
                        ) or is_llm_budget_exception(e):
                            raise
                        logger.error(
                            "Error analyzing plots for node %s: %s",
                            child_node.id,
                            type(e).__name__,
                        )

            _assert_preserved_evaluation_artifact(child_node)

            # Convert result node to dict
            print("Converting result to dict")
            result_data = child_node.to_dict()
            print(f"Result data keys: {result_data.keys()}")
            print(f"Result data size: {len(str(result_data))} chars")
            print("Returning result")
            return result_data

        except Exception as e:
            logger.error("Worker process failed: %s", type(e).__name__)
            raise
        finally:
            experiment_interpreter.cleanup_session()
            if analysis_interpreter is not None:
                analysis_interpreter.cleanup_session()

    def _generate_hyperparam_tuning_idea(
        self,
        excluded_keys: set[str] | None = None,
    ) -> Optional[HyperparamTuningIdea]:
        """Generate the next hyperparam tuning idea based on what's been done.
        This is minaly for Stage 2 (baseline tuning).
        """
        tried = sorted(
            set(self._hyperparam_tuning_state["tried_hyperparams"])
            | set(excluded_keys or ())
        )

        hyperparam_tuning_prompt = {
            "Introduction": (
                "You are an AI researcher conducting hyperparameter tuning for baseline experiments. "
                "Based on the current implementation and previous hyperparameter tuning attempts (if any), "
                "propose ONE new hyperparameter tuning idea to see if it improves the performance."
                "You should first check if simply training longer (more epochs) improves the performance."
                "Then try tuning common hyperparameters such as learning rate, batch size, etc."
                "Only propose algorithm-specific and/or model-specific hyperparameters after you have tried the above."
            ),
            "Base code you are working on": wrap_code(self.best_stage1_node.code),
            "Previous Hyperparam Tuning Attempts": {
                "Has been tried": tried if tried else "Nothing has been tried yet.",
            },
            "Instructions": {
                "Requirements": [
                    "1. Identify ONE specific hyperparameter to tune",
                    "2. Ensure the hyperparameter is different from previous attempts",
                ]
            },
            "Response format": (
                "Your response should start with 'HYPERPARAM NAME: <hyperparam name>' on the first line to represent the name of the hyperparameter."
                "The second line should start with 'DESCRIPTION: <description>', a brief description of what hyperparameter is being tuned and why (3-5 sentences). "
            ),
        }

        with capture_llm_calls() as refs:
            response = query(
                system_message=hyperparam_tuning_prompt,
                user_message=None,
                func_spec=hyperparam_idea_spec,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )
        return HyperparamTuningIdea(
            name=response["name"],
            description=response["description"],
            llm_call_refs=list(refs),
        )

    def _generate_ablation_idea(
        self,
        excluded_keys: set[str] | None = None,
    ) -> Optional[AblationIdea]:
        """Generate the next ablation idea based on what's been done"""

        # Prepare context of what's been tried
        completed = sorted(
            set(self._ablation_state["completed_ablations"]) | set(excluded_keys or ())
        )

        ablation_prompt = {
            "Introduction": (
                "You are an AI researcher conducting ablation studies. "
                "Based on the current implementation and previous ablations (if any), "
                "propose ONE new ablation study that tests a different aspect of the model."
            ),
            "Base code you are working on": wrap_code(self.best_stage3_node.code),
            "Previous Ablations": {
                "Has been tried": (
                    completed if completed else "Nothing has been tried yet."
                ),
            },
            "Instructions": {
                "Requirements": [
                    "1. Identify ONE specific component/feature to ablate",
                    "2. Ensure the ablation is different from previous completed or running attempts",
                    "3. The ablation should be a new idea, not a variation of previous ideas",
                    "4. If you have only used a single synthetic dataset throughout the experiment, one of your ablations should be to use multiple synthetic datasets (at least 3 different datasets)",
                ]
            },
        }
        with capture_llm_calls() as refs:
            response = query(
                system_message=ablation_prompt,
                user_message=None,
                func_spec=ablation_idea_spec,
                model=self.cfg.agent.code.model,
                temperature=self.cfg.agent.code.temp,
            )
        return AblationIdea(
            name=response["name"],
            component=response["component"],
            description=response["description"],
            expected_outcome=response["expected_outcome"],
            llm_call_refs=list(refs),
        )

    def _get_leaves(self, node: Node) -> List[Node]:
        """Get all leaf nodes in the subtree rooted at node."""
        if not node.children:
            return [node]

        leaves = []
        for child in node.children:
            leaves.extend(self._get_leaves(child))
        return leaves

    def _select_parallel_nodes(self) -> List[Optional[Node]]:
        """Select N nodes to process in parallel,
        balancing between tree exploration and exploitation.
        Note:
        - This function runs in the main process.
        Some design considerations:
        - For Stage 2 and 4, we generate nodes in the main process and
        send them to worker processes.
        This is to make sure we don't run duplicate ideas in parallel.
        - For Stage 1 and 3, we generate nodes in worker processes.
        """
        nodes_to_process = []
        processed_trees = set()
        search_cfg = self.cfg.agent.search
        search_mode = (
            str(os.environ.get("AI_SCIENTIST_SEARCH_MODE") or "").strip().lower()
        )
        reference_contract = None
        if (
            self.stage_name
            and not self.stage_name.startswith("1_")
            and self.journal.nodes
        ):
            reference_contract = self.journal.nodes[0].evaluation_comparison_contract
        print(f"[cyan]self.num_workers: {self.num_workers}, [/cyan]")

        while len(nodes_to_process) < self.num_workers:
            # Initial drafting phase, creating root nodes
            print(
                f"Checking draft nodes... num of journal.draft_nodes: {len(self.journal.draft_nodes)}, search_cfg.num_drafts: {search_cfg.num_drafts}"
            )
            if len(self.journal.draft_nodes) < search_cfg.num_drafts:
                nodes_to_process.append(None)
                continue

            # Get viable trees
            viable_trees = [
                root
                for root in self.journal.draft_nodes
                if not all(leaf.is_buggy for leaf in self._get_leaves(root))
            ]

            # Debugging phase (with some probability)
            if random.random() < search_cfg.debug_prob:
                print("Checking debuggable nodes")
                # print(f"Buggy nodes: {self.journal.buggy_nodes}")
                try:
                    debuggable_nodes = None
                    print("Checking buggy nodes...")
                    buggy_nodes = self.journal.buggy_nodes
                    print(f"Type of buggy_nodes: {type(buggy_nodes)}")
                    print(f"Length of buggy_nodes: {len(buggy_nodes)}")

                    for i, n in enumerate(buggy_nodes):
                        if not isinstance(n, Node):
                            print(f"Found non-Node object in journal.buggy_nodes: {n}")
                            raise ValueError(
                                "Found non-Node object in journal.buggy_nodes"
                            )
                    debuggable_nodes = [
                        n
                        for n in self.journal.buggy_nodes
                        if (
                            isinstance(n, Node)
                            and n.is_leaf
                            and n.debug_depth <= search_cfg.max_debug_depth
                        )
                    ]
                except Exception as e:
                    print(f"Error getting debuggable nodes: {type(e).__name__}")
                if debuggable_nodes:
                    print("Found debuggable nodes")
                    node = random.choice(debuggable_nodes)
                    tree_root = node
                    while tree_root.parent:
                        tree_root = tree_root.parent

                    tree_id = id(tree_root)
                    if tree_id not in processed_trees or len(processed_trees) >= len(
                        viable_trees
                    ):
                        nodes_to_process.append(node)
                        processed_trees.add(tree_id)
                        continue

            # Special handling for Stage 4 (Ablation Studies)
            print(f"[red]self.stage_name: {self.stage_name}[/red]")
            # print(f"[red]self.best_stage3_node: {self.best_stage3_node}[/red]")
            if self.stage_name and self.stage_name.startswith("4_"):
                nodes_to_process.append(self.best_stage3_node)
                continue
            # Special handling for Stage 2 (Hyperparam tuning for baseline)
            elif self.stage_name and self.stage_name.startswith("2_"):
                nodes_to_process.append(self.best_stage1_node)
                continue
            else:  # Stage 1, 3 (normal best-first search)
                # Improvement phase
                print("Checking good nodes..")
                good_nodes = [
                    node
                    for node in self.journal.verified_nodes
                    if reference_contract is None
                    or node.evaluation_comparison_contract == reference_contract
                ]
                if not good_nodes:
                    nodes_to_process.append(None)  # Back to drafting
                    continue

                # Autoresearch-style exploitation: always propose variants off the current best.
                # This mimics a keep/discard hill-climb loop (see karpathy/autoresearch).
                if search_mode == "autoresearch":
                    best_node = self.journal.get_best_node_by_metric(
                        reference_contract=reference_contract
                    )
                    if best_node is None:
                        nodes_to_process.append(None)
                        continue
                    nodes_to_process.append(best_node)
                    continue

                # Get best node deterministically (avoid extra LLM calls in the inner loop).
                best_node = self.journal.get_best_node_by_metric(
                    reference_contract=reference_contract
                )
                if best_node is None:
                    nodes_to_process.append(None)
                    continue
                tree_root = best_node
                while tree_root.parent:
                    tree_root = tree_root.parent

                tree_id = id(tree_root)
                if tree_id not in processed_trees or len(processed_trees) >= len(
                    viable_trees
                ):
                    nodes_to_process.append(best_node)
                    processed_trees.add(tree_id)
                    continue

                # If we can't use best node (tree already processed), try next best nodes
                for node in sorted(good_nodes, key=lambda n: n.metric, reverse=True):
                    tree_root = node
                    while tree_root.parent:
                        tree_root = tree_root.parent
                    tree_id = id(tree_root)
                    if tree_id not in processed_trees or len(processed_trees) >= len(
                        viable_trees
                    ):
                        nodes_to_process.append(node)
                        processed_trees.add(tree_id)
                        break

        return nodes_to_process

    def step(
        self,
        exec_callback: ExecCallbackType,
        *,
        max_new_nodes: int | None = None,
    ) -> int:
        if max_new_nodes is None:
            max_new_nodes = self.num_workers
        if (
            isinstance(max_new_nodes, bool)
            or not isinstance(max_new_nodes, int)
            or not 1 <= max_new_nodes <= self.num_workers
        ):
            raise ValueError("max_new_nodes must be between 1 and num_workers")
        print("Selecting nodes to process")
        nodes_to_process = self._select_parallel_nodes()[:max_new_nodes]
        print(f"Selected nodes: {[n.id if n else None for n in nodes_to_process]}")

        # Convert nodes to dicts
        node_data_list = []
        for node in nodes_to_process:
            if node:
                try:
                    node_data = node.to_dict()
                    _safe_pickle_test(node_data, f"node {node.id} data")
                    node_data_list.append(node_data)
                except Exception as e:
                    logger.error(
                        "Error preparing node %s: %s", node.id, type(e).__name__
                    )
                    raise
            else:
                node_data_list.append(None)  # None means new draft

        summary_cfg = (
            self.cfg.agent.summary
            if self.cfg.agent.get("summary", None) is not None
            else self.cfg.report
        )
        memory_summary = self.journal.generate_summary(
            include_code=False,
            model=summary_cfg.model,
            temp=summary_cfg.temp,
        )

        print("Submitting tasks to process pool")
        futures: list[Any] = []
        future_idea_keys: dict[Any, tuple[str, str] | None] = {}
        reserved_hyperparams = set(self._hyperparam_tuning_state["tried_hyperparams"])
        reserved_ablations = set(self._ablation_state["completed_ablations"])
        live_nodes = [node.to_dict() for node in self.journal.nodes]
        try:
            for node_data in node_data_list:
                worker_memory_summary = memory_summary
                context_pack_ref = None
                try:
                    from ai_scientist.utils.ara_context import (
                        compile_live_continue_context,
                        persist_active_context_pack,
                        render_context_pack_for_prompt,
                    )

                    context_pack = compile_live_continue_context(
                        live_nodes,
                        target_node_id=(
                            str(node_data.get("id")) if node_data else None
                        ),
                        stage=self.stage_name,
                        budget_tokens=3000,
                    )
                    context_pack_ref = persist_active_context_pack(
                        context_pack,
                        consumer="experiment_agent",
                    )
                    context_prompt = render_context_pack_for_prompt(context_pack)
                    legacy_summary = str(memory_summary or "")[:6000]
                    worker_memory_summary = (
                        f"{context_prompt}\n\n## Secondary journal synopsis\n{legacy_summary}"
                        if legacy_summary
                        else context_prompt
                    )
                except Exception as exc:
                    logger.warning(
                        "Could not compile live ARA context: %s",
                        type(exc).__name__,
                    )
                    legacy_summary = str(memory_summary or "")[:6000]
                    worker_memory_summary = (
                        "## Context fallback (degraded, not source-bound)\n"
                        f"reason={type(exc).__name__}\n"
                        "The semantic ContextPack was unavailable. Treat this journal "
                        "synopsis as advisory and do not promote claims from it.\n\n"
                        f"{legacy_summary}"
                    )

                new_ablation_idea = None
                new_hyperparam_idea = None
                expected_idea: tuple[str, str] | None = None
                if (
                    self.stage_name
                    and self.stage_name.startswith("2_")
                    and node_data is not None
                    and node_data["is_buggy"] is False
                ):
                    for _ in range(3):
                        proposal = self._generate_hyperparam_tuning_idea(
                            reserved_hyperparams
                        )
                        proposal_key = _canonical_idea_key(
                            proposal.name if proposal is not None else None
                        )
                        if proposal_key and proposal_key not in reserved_hyperparams:
                            new_hyperparam_idea = proposal
                            reserved_hyperparams.add(proposal_key)
                            expected_idea = ("hyperparam", proposal_key)
                            break
                    if new_hyperparam_idea is None:
                        raise ResearchDecisionError(
                            "Research Agent repeated a hyperparameter idea"
                        )
                elif (
                    self.stage_name
                    and self.stage_name.startswith("4_")
                    and node_data is not None
                    and node_data["is_buggy"] is False
                ):
                    for _ in range(3):
                        proposal = self._generate_ablation_idea(reserved_ablations)
                        proposal_key = _ablation_idea_key(
                            proposal.name if proposal is not None else None,
                            proposal.component if proposal is not None else None,
                        )
                        if proposal_key and proposal_key not in reserved_ablations:
                            new_ablation_idea = proposal
                            reserved_ablations.add(proposal_key)
                            expected_idea = ("ablation", proposal_key)
                            break
                    if new_ablation_idea is None:
                        raise ResearchDecisionError(
                            "Research Agent repeated an ablation idea"
                        )

                gpu_id = None
                if self.gpu_manager is not None:
                    process_id = f"worker_{len(futures)}"
                    gpu_id = self.gpu_manager.acquire_gpu(process_id)
                    logger.info("Assigned a GPU to a research worker")

                best_stage1_plot_code = (
                    self.best_stage1_node.plot_code if self.best_stage1_node else None
                )
                best_stage2_plot_code = (
                    self.best_stage2_node.plot_code if self.best_stage2_node else None
                )
                best_stage3_plot_code = (
                    self.best_stage3_node.plot_code if self.best_stage3_node else None
                )
                future = self.executor.submit(
                    self._process_node_wrapper,
                    node_data,
                    self.task_desc,
                    self.cfg,
                    gpu_id,
                    worker_memory_summary,
                    self.evaluation_metrics,
                    self.stage_name,
                    new_ablation_idea,
                    new_hyperparam_idea,
                    best_stage1_plot_code,
                    best_stage2_plot_code,
                    best_stage3_plot_code,
                    False,
                    context_pack_ref,
                )
                futures.append(future)
                future_idea_keys[future] = expected_idea
        except BaseException as exc:
            for pending in futures:
                pending.cancel()
            self.cleanup()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
                raise
            raise ExperimentCannotContinueError(
                "Research worker batch could not be submitted atomically"
            ) from None

        # Treat one reserved worker batch as a transaction with one deadline.
        # A partial batch is not scientific evidence and is never journaled.
        print("Waiting for bounded worker batch")
        staged_results: list[dict[str, Any]] = []
        try:
            _done, unfinished = wait(
                futures,
                timeout=self.timeout,
                return_when=ALL_COMPLETED,
            )
            if unfinished:
                raise TimeoutError("worker batch deadline exceeded")
            seen_ids: set[str] = set()
            for future in futures:
                result_data = future.result()
                if not isinstance(result_data, dict):
                    raise ExperimentCannotContinueError(
                        "Research worker returned an invalid result"
                    )
                parent_id = result_data.get("parent_id")
                if (
                    parent_id is not None
                    and self.journal.get_node_by_id(parent_id) is None
                ):
                    raise ExperimentCannotContinueError(
                        "Research worker result references a missing parent"
                    )
                staged_node = Node.from_dict(copy.deepcopy(result_data))
                expected_idea = future_idea_keys.get(future)
                actual_idea = None
                if expected_idea is not None and expected_idea[0] == "hyperparam":
                    actual_idea = _canonical_idea_key(staged_node.hyperparam_name)
                elif expected_idea is not None and expected_idea[0] == "ablation":
                    actual_idea = _ablation_idea_key(
                        staged_node.ablation_name,
                        staged_node.ablation_component,
                    )
                if (
                    not isinstance(staged_node.id, str)
                    or not staged_node.id
                    or staged_node.id in seen_ids
                    or self.journal.get_node_by_id(staged_node.id) is not None
                    or staged_node.is_seed_node
                    or staged_node.is_seed_agg_node
                    or (expected_idea is not None and actual_idea != expected_idea[1])
                ):
                    raise ExperimentCannotContinueError(
                        "Research worker result identity is invalid"
                    )
                seen_ids.add(staged_node.id)
                staged_results.append(copy.deepcopy(result_data))
        except BaseException as exc:
            for pending in futures:
                pending.cancel()
            self.cleanup()
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, ResearchDecisionError) or is_llm_budget_exception(exc):
                raise
            logger.error("Research worker batch failed: %s", type(exc).__name__)
            raise ExperimentCannotContinueError(
                "Research worker batch did not complete atomically"
            ) from None
        finally:
            if self.gpu_manager is not None:
                for process_id in list(self.gpu_manager.gpu_assignments):
                    self.gpu_manager.release_gpu(process_id)

        journal_size = len(self.journal.nodes)
        parent_children = {node.id: set(node.children) for node in self.journal.nodes}
        tried_hyperparams = set(self._hyperparam_tuning_state["tried_hyperparams"])
        completed_ablations = set(self._ablation_state["completed_ablations"])
        committed_nodes: list[Node] = []
        try:
            for result_data in staged_results:
                result_node = Node.from_dict(copy.deepcopy(result_data), self.journal)
                self.journal.append(result_node)
                committed_nodes.append(result_node)
            for result_node in committed_nodes:
                self._update_hyperparam_tuning_state(result_node)
                self._update_ablation_state(result_node)
        except BaseException as exc:
            del self.journal.nodes[journal_size:]
            for node in self.journal.nodes:
                node.children = parent_children.get(node.id, set())
            self._hyperparam_tuning_state["tried_hyperparams"] = tried_hyperparams
            self._ablation_state["completed_ablations"] = completed_ablations
            if not isinstance(exc, Exception):
                raise
            raise ExperimentCannotContinueError(
                "Research worker batch could not be committed atomically"
            ) from None

        for result_node in committed_nodes:
            for ref in result_node.context_pack_refs or []:
                try:
                    from ai_scientist.utils.ara_context import (
                        record_active_context_consumption,
                    )

                    record_active_context_consumption(
                        pack_ref=ref,
                        consumer="experiment_agent",
                        output_type="node",
                        output_id=result_node.id,
                    )
                except Exception:
                    pass
        return len(committed_nodes)

    def _update_hyperparam_tuning_state(self, result_node: Node):
        """Update hyperparam tuning tracking state based on execution results."""
        if not self.stage_name or not self.stage_name.startswith("2_"):
            return

        hyperparam_name = result_node.hyperparam_name
        if hyperparam_name is None:
            print(
                f"[red]hyperparam_name is None for result_node: {result_node.id}[/red]"
            )
            return

        key = _canonical_idea_key(hyperparam_name)
        if not key:
            raise ExperimentCannotContinueError(
                "Hyperparameter attempt has an invalid identity"
            )
        self._hyperparam_tuning_state["tried_hyperparams"].add(key)
        logger.info("Recorded a hyperparameter attempt")

    def _update_ablation_state(self, result_node: Node):
        """Update ablation tracking state based on execution results.

        Args:
            result_node: Node containing ablation execution results
        """
        if not self.stage_name or not self.stage_name.startswith("4_"):
            return

        ablation_name = result_node.ablation_name
        if ablation_name is None:
            print(f"[red]ablation_name is None for result_node: {result_node.id}[/red]")
            return

        key = _ablation_idea_key(ablation_name, result_node.ablation_component)
        if not key:
            raise ExperimentCannotContinueError(
                "Ablation attempt has an invalid identity"
            )
        self._ablation_state["completed_ablations"].add(key)
        logger.info("Recorded an ablation attempt")

    def _aggregate_seed_eval_results(
        self, seed_nodes: List[Node], parent_node: Node
    ) -> str:
        """Generate aggregated plots from multi-seed evaluation results.

        Args:
            seed_nodes: List of nodes from seed evaluation
            parent_node: The original node that was evaluated

        Returns:
            str: The plotting code for aggregated results
        """
        prompt_guideline = []
        prompt_guideline += [
            "REQUIREMENTS: ",
            "The code should start with:",
            "  import matplotlib.pyplot as plt",
            "  import numpy as np",
            "  import os",
            "  working_dir = os.path.join(os.getcwd(), 'working')",
            "Create standard visualizations of experiment results",
            "Save all plots to working_dir",
            "Include training/validation curves if available",
            "ONLY plot data that exists in experiment_data.npy - DO NOT make up or simulate any values",
            "Use basic matplotlib without custom styles",
            "Each plot should be in a separate try-except block",
            "Always close figures after saving",
            "Always include a title for each plot, and be sure to use clear subtitles—such as 'Left: Ground Truth, Right: Generated Samples'—while also specifying the type of dataset being used.",
            "Make sure to use descriptive names for figures when saving e.g. always include the dataset name and the type of plot in the name",
            "When there are many similar figures to plot (e.g. generated samples at each epoch), make sure to plot only at a suitable interval of epochs so that you only plot at most 5 figures.",
            "Example to extract data from experiment_data: experiment_data['dataset_name_1']['metrics']['train']",
            "Make sure to add legend for standard error bars and means if applicable",
        ]
        prompt_guideline += [
            "Example data loading and plot saving code: ",
            """
                try:
                    experiment_data_path_list = # Make sure to use the correct experiment data path that's provided in the Experiment Data Path section
                    all_experiment_data = []
                    for experiment_data_path in experiment_data_path_list:
                        experiment_data = np.load(experiment_data_path, allow_pickle=True).item()
                        all_experiment_data.append(experiment_data)
                except Exception as e:
                    print(f'Error loading experiment data: {{e}}')

                try:
                    # First plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_1].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot1: {{e}}")
                    plt.close()  # Always close figure even if error occurs

                try:
                    # Second plot
                    plt.figure()
                    # ... plotting code ...
                    plt.savefig('working_dir/[plot_name_2].png')
                    plt.close()
                except Exception as e:
                    print(f"Error creating plot2: {{e}}")
                    plt.close()
            """,
        ]
        # add instruction for format
        plotting_prompt = {
            "Introduction": (
                "You are an expert in data visualization and plotting. "
                "You are given a set of evaluation results and the code that was used to plot them. "
                "Your task is to write a new plotting code that aggregate the results "
                "e.g. for example, by adding mean values and standard error bars to the plots."
            ),
            "Instructions": {},
        }
        plotting_prompt["Instructions"] |= {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (7-10 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric(s) if applicable. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }
        plotting_prompt["Instructions"] |= {
            "Plotting code guideline": prompt_guideline,
        }
        if any(
            not isinstance(seed.plot_code, str)
            or not seed.plot_code.strip()
            or not isinstance(seed.exp_results_dir, (str, Path))
            for seed in seed_nodes
        ):
            raise ExperimentCannotContinueError(
                "Verified seed evidence is missing plotting artifacts"
            )
        plotting_prompt["Instructions"] |= {
            "Plotting code reference": "\n\n".join(
                f"plotting code {index}:\n{seed.plot_code}"
                for index, seed in enumerate(seed_nodes, start=1)
            ),
            "Experiment Data Path": "\n".join(
                f"{seed.exp_results_dir}/experiment_data.npy" for seed in seed_nodes
            ),
        }
        plan, code = self.plan_and_code_query(plotting_prompt)

        logger.info(
            "Generated seed aggregation plan/code: plan_chars=%d code_chars=%d",
            len(plan),
            len(code),
        )

        return code

    def __enter__(self):
        return self

    def cleanup(self):
        """Cleanup parallel workers and resources"""
        if not self._is_shutdown:
            print("Shutting down parallel executor...")
            try:
                # Release all GPUs
                if self.gpu_manager is not None:
                    for process_id in list(self.gpu_manager.gpu_assignments.keys()):
                        self.gpu_manager.release_gpu(process_id)

                executor = getattr(self, "executor", None)
                processes = list((getattr(executor, "_processes", None) or {}).values())
                if executor is not None:
                    executor.shutdown(wait=False, cancel_futures=True)

                # Force terminate all worker processes
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join(timeout=1)

                print("Executor shutdown complete")

            except Exception as e:
                print(f"Error during executor shutdown: {type(e).__name__}")
            finally:
                self._is_shutdown = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
