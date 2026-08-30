# Modified by XScientist contributors from the AI-Scientist-v2/AIDE lineage.
# See THIRD_PARTY_NOTICES.md for provenance and license details.
from typing import List, Optional, Dict, Callable, Any, Mapping, Tuple
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import logging
import hashlib
import math
import os
import stat
import statistics
from .parallel_agent import (
    ParallelAgent,
    _assert_glm53_authority_boundary,
    _ablation_component_was_transformed,
    _configured_multi_seed_values,
    _inject_seed_bootstrap,
    _is_glm53_stage,
    _semantic_code_hash,
    _stage_max_tokens,
    _validate_confirmation_seed_set,
)
from .errors import ExperimentCannotContinueError, MultiSeedGateRejectedError
from .journal import Journal, Node
import copy
import re
from .backend import (
    FunctionCallValidationError,
    FunctionSpec,
    ResearchDecisionError,
    query,
)
import json
from rich import print
from .utils.serialize import atomic_write_json, parse_markdown_to_dict
from .utils.metric import WorstMetricValue
from ai_scientist.utils.llm_budget import is_llm_budget_exception
from ai_scientist.utils.privacy import portable_path
from ai_scientist.utils.evaluation_binding import evaluation_hash_binding
from ai_scientist.utils.deterministic_evaluator import evaluate_experiment_data
from ai_scientist.utils.authority_attempts import (
    AuthorityAttemptError,
    inspect_authority_attempts,
)

logger = logging.getLogger(__name__)

CHECKPOINT_SCHEMA = "xscientist.bfts.checkpoint.v5"
INPUT_TREE_SCHEMA = "xscientist.bfts.input-tree.v1"
INPUT_TREE_MAX_ENTRIES = 100_000
INPUT_TREE_MAX_BYTES = 10 * 1024 * 1024 * 1024
INPUT_TREE_MAX_DEPTH = 256
STAGE_NAME_COMPONENT_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
STAGE_NAME_COMPONENT_RE = re.compile(STAGE_NAME_COMPONENT_PATTERN)
MAX_AGENT_STAGE_ITERATIONS = 100
MAIN_STAGE_NAMES = {
    1: "initial_implementation",
    2: "baseline_tuning",
    3: "creative_research",
    4: "ablation_studies",
}
STUDENT_T_95_CRITICAL_BY_DF = {
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    31: 2.040,
}


def _validate_restored_authority_ledger(
    journals: Mapping[str, Journal],
    *,
    log_dir: Path,
) -> None:
    """Fail closed when checkpointed Node commitments and ledger diverge."""

    expected_hashes: dict[str, str] = {}
    for journal in journals.values():
        for node in journal.nodes:
            attempt_ids = list(node.authority_attempt_ids or [])
            terminal_hashes = dict(node.authority_attempt_terminal_hashes or {})
            if set(terminal_hashes) != set(attempt_ids):
                raise ValueError(
                    "BFTS checkpoint contains incomplete authority attempt bindings"
                )
            for attempt_id in attempt_ids:
                event_hash = terminal_hashes[attempt_id]
                prior = expected_hashes.get(attempt_id)
                if prior is not None and prior != event_hash:
                    raise ValueError(
                        "BFTS checkpoint contains conflicting authority terminal hashes"
                    )
                expected_hashes[attempt_id] = event_hash

    attempt_root = log_dir / "authority_attempts"
    ledger_present = os.path.lexists(attempt_root)
    if not expected_hashes and not ledger_present:
        return
    try:
        audit = inspect_authority_attempts(
            log_dir,
            expected_attempt_ids=expected_hashes,
        )
    except AuthorityAttemptError as exc:
        raise ValueError("BFTS checkpoint authority attempt ledger is invalid") from exc
    if not audit.get("valid") or not audit.get("expected_valid"):
        raise ValueError(
            "BFTS checkpoint authority attempt ledger is incomplete, orphaned, "
            "tampered, or missing"
        )
    audited_hashes = audit.get("terminal_event_hashes") or {}
    if any(
        audited_hashes.get(attempt_id) != event_hash
        for attempt_id, event_hash in expected_hashes.items()
    ):
        raise ValueError(
            "BFTS checkpoint authority terminal hashes do not match the ledger"
        )


def _student_t_95_critical(sample_count: int) -> float:
    if isinstance(sample_count, bool) or not 3 <= sample_count <= 32:
        raise ExperimentCannotContinueError(
            "Student-t confidence interval requires 3-32 samples"
        )
    return STUDENT_T_95_CRITICAL_BY_DF[sample_count - 1]


def _parse_stage_name(stage_name: str) -> Tuple[int, str, int, str]:
    main_text, separator, remainder = str(stage_name).partition("_")
    if not separator or not main_text.isdigit():
        raise ValueError("Invalid stage name")
    main_stage = int(main_text)
    main_stage_name = MAIN_STAGE_NAMES.get(main_stage)
    if main_stage_name is None:
        raise ValueError("Invalid main stage number")
    expected_prefix = main_stage_name + "_"
    if not remainder.startswith(expected_prefix):
        raise ValueError("Invalid main stage name")
    substage_text = remainder[len(expected_prefix) :]
    sub_stage_number_text, separator, sub_stage_name = substage_text.partition("_")
    if (
        not separator
        or not sub_stage_number_text.isdigit()
        or not STAGE_NAME_COMPONENT_RE.fullmatch(sub_stage_name)
    ):
        raise ValueError("Invalid sub-stage name")
    return (
        main_stage,
        main_stage_name,
        int(sub_stage_number_text),
        sub_stage_name,
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _sha256_json(value: Any) -> str:
    return (
        "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()
    )


def _strict_sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _checkpoint_config_projection(cfg: Any) -> dict:
    def plain(value: Any) -> Any:
        if hasattr(value, "items"):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    return {
        "agent": plain(cfg.agent),
        "exec": plain(cfg.exec),
        "experiment": plain(cfg.experiment),
        "debug": plain(cfg.debug),
    }


def _input_tree_manifest(workspace_dir: Path) -> dict:
    """Build a portable, content-addressed manifest for ``workspace/input``.

    The manifest exists only in memory. Checkpoints persist its digest, not
    absolute paths, file names, link targets, or file contents. Symlinks are
    followed so the default linked-data workspace and a copied workspace have
    the same identity when their effective inputs are identical.
    """

    input_dir = Path(workspace_dir) / "input"
    entries: list[dict[str, Any]] = []
    total_bytes = 0

    def fail(reason: str) -> None:
        raise ValueError(f"Cannot safely fingerprint BFTS input tree: {reason}")

    def stable_stat_fields(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def encoded_name(path: Path) -> str:
        # Hex encoding makes even surrogate-escaped POSIX names canonical and
        # avoids retaining readable input names in the in-memory manifest.
        return os.fsencode(path.name).hex()

    def add_entry(entry: dict[str, Any]) -> None:
        if len(entries) >= INPUT_TREE_MAX_ENTRIES:
            fail(f"entry limit ({INPUT_TREE_MAX_ENTRIES}) exceeded")
        entries.append(entry)

    def hash_file(path: Path, initial_stat: os.stat_result) -> str:
        nonlocal total_bytes
        if total_bytes + initial_stat.st_size > INPUT_TREE_MAX_BYTES:
            fail(f"byte limit ({INPUT_TREE_MAX_BYTES}) exceeded")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            final_stat = path.stat()
        except OSError as exc:
            raise ValueError(
                "Cannot safely fingerprint BFTS input tree: unreadable file"
            ) from exc
        if stable_stat_fields(initial_stat) != stable_stat_fields(final_stat):
            fail("file changed during traversal")
        total_bytes += final_stat.st_size
        return "sha256:" + digest.hexdigest()

    def walk_directory(
        directory: Path,
        relative_parts: tuple[str, ...],
        ancestor_directories: frozenset[tuple[int, int]],
    ) -> None:
        if len(relative_parts) > INPUT_TREE_MAX_DEPTH:
            fail(f"depth limit ({INPUT_TREE_MAX_DEPTH}) exceeded")
        try:
            directory_before = directory.stat()
            directory_link_before = directory.lstat()
        except OSError as exc:
            raise ValueError(
                "Cannot safely fingerprint BFTS input tree: invalid directory link"
            ) from exc
        if not stat.S_ISDIR(directory_before.st_mode):
            fail("input root is not a directory")
        directory_identity = (directory_before.st_dev, directory_before.st_ino)
        if directory_identity in ancestor_directories:
            fail("directory symlink cycle detected")
        ancestors = ancestor_directories | {directory_identity}
        try:
            children = sorted(
                directory.iterdir(), key=lambda child: os.fsencode(child.name)
            )
        except OSError as exc:
            raise ValueError(
                "Cannot safely fingerprint BFTS input tree: unreadable directory"
            ) from exc

        for child in children:
            child_parts = relative_parts + (encoded_name(child),)
            manifest_path = "/".join(child_parts)
            try:
                link_before = child.lstat()
                resolved_before = child.stat()
            except OSError as exc:
                raise ValueError(
                    "Cannot safely fingerprint BFTS input tree: broken or invalid link"
                ) from exc

            if stat.S_ISDIR(resolved_before.st_mode):
                add_entry({"path": manifest_path, "type": "directory"})
                walk_directory(child, child_parts, ancestors)
            elif stat.S_ISREG(resolved_before.st_mode):
                add_entry(
                    {
                        "path": manifest_path,
                        "type": "file",
                        "size": resolved_before.st_size,
                        "digest": hash_file(child, resolved_before),
                    }
                )
            else:
                fail("non-regular input entry encountered")

            try:
                link_after = child.lstat()
            except OSError as exc:
                raise ValueError(
                    "Cannot safely fingerprint BFTS input tree: entry changed during traversal"
                ) from exc
            if stable_stat_fields(link_before) != stable_stat_fields(link_after):
                fail("entry changed during traversal")

        try:
            directory_after = directory.stat()
            directory_link_after = directory.lstat()
        except OSError as exc:
            raise ValueError(
                "Cannot safely fingerprint BFTS input tree: directory changed during traversal"
            ) from exc
        if stable_stat_fields(directory_before) != stable_stat_fields(directory_after):
            fail("directory changed during traversal")
        if stable_stat_fields(directory_link_before) != stable_stat_fields(
            directory_link_after
        ):
            fail("directory link changed during traversal")

    try:
        input_dir.lstat()
    except FileNotFoundError:
        return {
            "schema": INPUT_TREE_SCHEMA,
            "root": "missing",
            "entries": [],
            "total_file_bytes": 0,
        }
    except OSError as exc:
        raise ValueError(
            "Cannot safely fingerprint BFTS input tree: inaccessible input root"
        ) from exc

    walk_directory(input_dir, (), frozenset())
    return {
        "schema": INPUT_TREE_SCHEMA,
        "root": "directory",
        "entries": entries,
        "total_file_bytes": total_bytes,
    }


def _input_tree_identity(workspace_dir: Path) -> dict[str, str]:
    manifest = _input_tree_manifest(workspace_dir)
    return {
        "schema": INPUT_TREE_SCHEMA,
        "fingerprint": _sha256_json(manifest),
    }


stage_config_spec = FunctionSpec(
    name="generate_stage_config",
    description="Generate configuration for the next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "pattern": STAGE_NAME_COMPONENT_PATTERN,
                "description": "Brief, descriptive name for the stage",
            },
            "description": {
                "type": "string",
                "description": "Detailed description of the stage's purpose",
            },
            "goals": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of specific, measurable goals for this stage",
            },
            "max_iterations": {
                "type": "integer",
                "minimum": 1,
                "maximum": MAX_AGENT_STAGE_ITERATIONS,
                "description": "Maximum number of iterations to run in this stage",
            },
        },
        "required": ["name", "description", "goals", "max_iterations"],
        "additionalProperties": False,
    },
)

stage_progress_eval_spec = FunctionSpec(
    name="evaluate_stage_progression",
    description="Evaluate readiness to progress to next experimental stage",
    json_schema={
        "type": "object",
        "properties": {
            "ready_for_next_stage": {
                "type": "boolean",
                "description": "Whether the experiment is ready to progress to next stage",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the progression decision",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific recommendations for current or next stage",
            },
            "suggested_focus": {
                "type": "string",
                "description": "Key areas to focus on in the next iterations",
            },
        },
        "required": ["ready_for_next_stage", "reasoning", "recommendations"],
        "additionalProperties": False,
    },
)


stage_completion_eval_spec = FunctionSpec(
    name="evaluate_stage_completion",
    description="Evaluate if the current stage is complete",
    json_schema={
        "type": "object",
        "properties": {
            "is_complete": {
                "type": "boolean",
                "description": "Whether the current stage is complete",
            },
            "reasoning": {
                "type": "string",
                "description": "Detailed reasoning for the decision",
            },
            "missing_criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of criteria still needed",
            },
        },
        "required": ["is_complete", "reasoning", "missing_criteria"],
        "additionalProperties": False,
    },
)


@dataclass
class Stage:
    name: str
    description: str
    goals: str | List[str]
    max_iterations: int
    num_drafts: int
    stage_number: int
    attempt_count: int = 0
    evaluation_metric: str | None = None
    qualified_node_id: str | None = None
    multi_seed_receipt_hash: str | None = None


@dataclass
class StageTransition:
    """Records transition between stages and the reasoning"""

    from_stage: str
    to_stage: str
    reason: str
    config_adjustments: Dict[str, Any]


def _fail_research_decision(context: str, error: BaseException) -> None:
    """Stop the resumable run without persisting provider-controlled text."""

    logger.error("Research decision failed in %s: %s", context, type(error).__name__)
    raise ResearchDecisionError(f"Research decision failed in {context}") from None


def _validate_stage_completion_evaluation(evaluation: Mapping[str, Any]) -> None:
    is_complete = evaluation.get("is_complete") is True
    missing = evaluation.get("missing_criteria")
    if not isinstance(missing, list) or any(
        not isinstance(item, str) or not item.strip() for item in missing
    ):
        raise FunctionCallValidationError(
            "Stage completion missing criteria are invalid"
        )
    if is_complete == bool(missing):
        raise FunctionCallValidationError(
            "Stage completion status conflicts with missing criteria"
        )


def _require_stage_progression(evaluation: Mapping[str, Any]) -> None:
    if evaluation.get("ready_for_next_stage") is not True:
        logger.warning("Research agent declined stage progression")
        raise ResearchDecisionError("Research agent declined stage progression")


def _validate_stage_definition(stage: Stage) -> Stage:
    parsed_stage_number, _, _, _ = _parse_stage_name(stage.name)
    if (
        isinstance(stage.stage_number, bool)
        or not isinstance(stage.stage_number, int)
        or stage.stage_number not in MAIN_STAGE_NAMES
        or parsed_stage_number != stage.stage_number
    ):
        raise ValueError("Stage number is invalid")
    if (
        isinstance(stage.max_iterations, bool)
        or not isinstance(stage.max_iterations, int)
        or not 1 <= stage.max_iterations <= MAX_AGENT_STAGE_ITERATIONS
    ):
        raise ValueError("Stage iteration budget is invalid")
    if (
        isinstance(stage.num_drafts, bool)
        or not isinstance(stage.num_drafts, int)
        or not 0 <= stage.num_drafts <= MAX_AGENT_STAGE_ITERATIONS
    ):
        raise ValueError("Stage draft budget is invalid")
    if (
        isinstance(stage.attempt_count, bool)
        or not isinstance(stage.attempt_count, int)
        or not 0 <= stage.attempt_count <= stage.max_iterations
    ):
        raise ValueError("Stage attempt count is invalid")
    if not isinstance(stage.description, str) or not stage.description.strip():
        raise ValueError("Stage description is invalid")
    if isinstance(stage.goals, str):
        goals_valid = bool(stage.goals.strip()) and len(stage.goals) <= 20_000
    elif isinstance(stage.goals, list):
        goals_valid = (
            bool(stage.goals)
            and len(stage.goals) <= 64
            and all(
                isinstance(goal, str) and bool(goal.strip()) and len(goal) <= 2_000
                for goal in stage.goals
            )
        )
    else:
        goals_valid = False
    if not goals_valid:
        raise ValueError("Stage goals are invalid")
    if stage.qualified_node_id is not None and (
        not isinstance(stage.qualified_node_id, str)
        or re.fullmatch(r"[A-Za-z0-9._-]{1,128}", stage.qualified_node_id) is None
    ):
        raise ValueError("Stage qualified node id is invalid")
    if stage.evaluation_metric is not None and (
        not isinstance(stage.evaluation_metric, str)
        or not stage.evaluation_metric.strip()
        or len(stage.evaluation_metric) > 2_000
    ):
        raise ValueError("Stage evaluation metric is invalid")
    if stage.multi_seed_receipt_hash is not None and (
        not isinstance(stage.multi_seed_receipt_hash, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", stage.multi_seed_receipt_hash) is None
    ):
        raise ValueError("Stage multi-seed receipt hash is invalid")
    return stage


def _evaluations_are_comparable(baseline: Node, candidate: Node) -> bool:
    baseline_contract = baseline.evaluation_comparison_contract
    candidate_contract = candidate.evaluation_comparison_contract
    if baseline_contract is None or candidate_contract is None:
        return False
    return (
        bool(baseline_contract["datasets"]) and baseline_contract == candidate_contract
    )


def _metric_improves_on_baseline(
    baseline: Node,
    candidate: Node,
) -> bool:
    return _metric_improvement_count(baseline, candidate) > 0


def _metric_improvement_count(baseline: Node, candidate: Node) -> int:
    """Count locked datasets that improve; return -1 on any regression."""

    if not baseline.has_verified_metric or not candidate.has_verified_metric:
        return -1
    if baseline.metric.comparison_family != candidate.metric.comparison_family:
        return -1
    if not _evaluations_are_comparable(baseline, candidate):
        return -1
    baseline_values = baseline.metric.values_by_dataset()
    candidate_values = candidate.metric.values_by_dataset()
    baseline_datasets = set(baseline_values)
    if not baseline_datasets or not baseline_datasets.issubset(candidate_values):
        return -1
    improves = 0
    maximize = candidate.metric._should_maximize()
    for name in baseline_datasets:
        baseline_value = baseline_values[name]
        candidate_value = candidate_values[name]
        tolerance = max(1e-9, abs(baseline_value) * 1e-6)
        signed_delta = (
            candidate_value - baseline_value
            if maximize
            else baseline_value - candidate_value
        )
        if signed_delta < -tolerance:
            return -1
        if signed_delta > tolerance:
            improves += 1
    return improves


def _qualified_candidate_score(baseline: Node, candidate: Node) -> tuple[float, float]:
    """Rank exact-contract candidates by their weakest and total improvement."""

    baseline_values = baseline.metric.values_by_dataset()
    candidate_values = candidate.metric.values_by_dataset()
    maximize = candidate.metric._should_maximize()
    deltas = [
        (
            candidate_values[name] - baseline_values[name]
            if maximize
            else baseline_values[name] - candidate_values[name]
        )
        for name in baseline_values
    ]
    return min(deltas), sum(deltas)


def _candidate_was_rejected(node: Node, stage_name: str) -> bool:
    return any(
        isinstance(attempt, dict) and attempt.get("stage") == stage_name
        for attempt in node.multi_seed_attempts
    )


def _ablation_code_diff_hash(control_code: str, ablation_code: str) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            _canonical_json(
                {
                    "control_code": control_code.strip(),
                    "ablation_code": ablation_code.strip(),
                }
            ).encode("utf-8")
        ).hexdigest()
    )


def _validate_multi_seed_report(node: Node, report: Any) -> dict[str, Any]:
    if (
        not isinstance(report, dict)
        or report.get("schema") != "xscientist.multi-seed.v1"
    ):
        raise ExperimentCannotContinueError("Multi-seed receipt is invalid")
    payload = dict(report)
    receipt_hash = payload.pop("receipt_hash", None)
    if receipt_hash != _strict_sha256_json(payload):
        raise ExperimentCannotContinueError("Multi-seed receipt hash mismatch")
    code_hash = "sha256:" + hashlib.sha256(node.code.encode("utf-8")).hexdigest()
    seeds = report.get("seeds")
    configured_seeds = report.get("configured_seeds")
    seed_node_ids = (
        [item.get("node_id") for item in seeds if isinstance(item, dict)]
        if isinstance(seeds, list)
        else []
    )
    control_receipt_hash = report.get("control_receipt_hash")
    if (
        report.get("qualified_node_id") != node.id
        or report.get("qualified_code_sha256") != code_hash
        or report.get("evaluation_contract") != node.evaluation_comparison_contract
        or not isinstance(report.get("stage"), str)
        or not report.get("stage")
        or not isinstance(seeds, list)
        or not isinstance(configured_seeds, list)
        or len(seeds) < 3
        or [item.get("seed") for item in seeds if isinstance(item, dict)]
        != configured_seeds
        or any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 2**32 - 1
            for seed in configured_seeds
        )
        or len(set(configured_seeds)) != len(configured_seeds)
        or len(seed_node_ids) != len(set(seed_node_ids))
        or (
            control_receipt_hash is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", str(control_receipt_hash)) is None
        )
    ):
        raise ExperimentCannotContinueError(
            "Multi-seed receipt does not bind the qualified node"
        )
    _validate_confirmation_seed_set(node.code, configured_seeds)
    datasets = report.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(
        node.metric.values_by_dataset()
    ):
        raise ExperimentCannotContinueError("Multi-seed dataset statistics are invalid")
    policy = report.get("stability_policy")
    if not isinstance(policy, dict):
        raise ExperimentCannotContinueError("Multi-seed stability policy is invalid")
    relative_limit = policy.get("max_relative_ci_half_width")
    absolute_floor = policy.get("absolute_ci_floor")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
        for value in (relative_limit, absolute_floor)
    ):
        raise ExperimentCannotContinueError("Multi-seed stability policy is invalid")
    recomputed_values: dict[str, list[float]] = {name: [] for name in datasets}
    for item in seeds:
        if (
            not isinstance(item, dict)
            or isinstance(item.get("seed"), bool)
            or not isinstance(item.get("seed"), int)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(item.get("seed_bootstrap_hash"))
            )
            is None
            or not isinstance(item.get("node_id"), str)
            or not isinstance(item.get("evaluation_binding"), dict)
            or not isinstance(item.get("values_by_dataset"), dict)
            or set(item["values_by_dataset"]) != set(datasets)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in item["values_by_dataset"].values()
            )
        ):
            raise ExperimentCannotContinueError("Multi-seed member receipt is invalid")
        for name, value in item["values_by_dataset"].items():
            recomputed_values[name].append(float(value))
    for name, stats in datasets.items():
        if not isinstance(stats, dict) or any(
            isinstance(stats.get(name), bool)
            or not isinstance(stats.get(name), (int, float))
            or not math.isfinite(float(stats[name]))
            for name in ("mean", "sample_stdev", "standard_error", "ci95_half_width")
        ):
            raise ExperimentCannotContinueError(
                "Multi-seed dataset statistics are invalid"
            )
        values = recomputed_values[name]
        expected_stdev = statistics.stdev(values)
        expected_error = expected_stdev / math.sqrt(len(values))
        expected_half_width = _student_t_95_critical(len(values)) * expected_error
        expected_mean = statistics.mean(values)
        expected_threshold = max(
            float(absolute_floor),
            abs(expected_mean) * float(relative_limit),
        )
        expected = {
            "n": len(values),
            "mean": expected_mean,
            "sample_stdev": expected_stdev,
            "standard_error": expected_error,
            "ci95_half_width": expected_half_width,
            "min": min(values),
            "max": max(values),
            "stability_threshold": expected_threshold,
            "stable": expected_half_width <= expected_threshold,
        }
        if stats != expected or stats["stable"] is not True:
            raise ExperimentCannotContinueError(
                "Multi-seed dataset statistics do not match seed evidence"
            )
    return report


def _validate_multi_seed_journal_links(
    journal: Journal,
    node: Node,
    report: Any,
) -> dict[str, Any]:
    """Resolve every receipt member back to immutable journal evidence."""

    validated = _validate_multi_seed_report(node, report)
    rejected_ids = _validate_rejected_multi_seed_attempts(
        journal,
        node,
        expected_stage_name=validated["stage"],
        expected_configured_seeds=validated["configured_seeds"],
    )
    reported_ids: set[str] = set()
    for row in validated["seeds"]:
        seed_node = journal.get_node_by_id(row["node_id"])
        expected_code, expected_bootstrap = _inject_seed_bootstrap(
            node.code,
            row["seed"],
        )
        if (
            seed_node is None
            or seed_node.id in reported_ids
            or seed_node.is_seed_node is not True
            or seed_node.is_seed_agg_node is True
            or seed_node.parent is None
            or seed_node.parent.id != node.id
            or seed_node.random_seed != row["seed"]
            or seed_node.code != expected_code
            or seed_node.seed_bootstrap_hash != expected_bootstrap
            or row.get("seed_bootstrap_hash") != expected_bootstrap
            or evaluation_hash_binding(seed_node.evaluation_report)
            != row["evaluation_binding"]
            or seed_node.evaluation_comparison_contract
            != node.evaluation_comparison_contract
            or seed_node.metric.values_by_dataset() != row["values_by_dataset"]
        ):
            raise ExperimentCannotContinueError(
                "Multi-seed receipt member does not resolve to journal evidence"
            )
        reported_ids.add(seed_node.id)
    actual_ids = {
        child.id
        for child in node.children
        if child.is_seed_node and not child.is_seed_agg_node
    }
    if actual_ids != reported_ids | rejected_ids:
        raise ExperimentCannotContinueError(
            "Multi-seed journal members do not match the receipt"
        )
    return validated


def _build_rejected_multi_seed_attempt(
    *,
    stage: Stage,
    node: Node,
    seed_nodes: list[Node],
    configured_seeds: list[int],
    reason_code: str,
    max_relative_ci_half_width: float,
    absolute_ci_floor: float,
    control_report: dict[str, Any] | None,
    required_improvements: int | None,
) -> dict[str, Any]:
    if reason_code not in {"stability", "paired_improvement"}:
        raise ExperimentCannotContinueError("Multi-seed rejection reason is invalid")
    contract = node.evaluation_comparison_contract
    if contract is None or len(seed_nodes) != len(configured_seeds):
        raise ExperimentCannotContinueError(
            "Rejected multi-seed evidence is incomplete"
        )
    rows = []
    values_by_dataset: dict[str, list[float]] = {
        name: [] for name in node.metric.values_by_dataset()
    }
    for expected_seed, seed_node in zip(configured_seeds, seed_nodes):
        binding = evaluation_hash_binding(seed_node.evaluation_report)
        if (
            seed_node.random_seed != expected_seed
            or not seed_node.has_verified_metric
            or seed_node.evaluation_comparison_contract != contract
            or binding is None
        ):
            raise ExperimentCannotContinueError(
                "Rejected seed evidence does not match its candidate"
            )
        rows.append(
            {
                "seed": expected_seed,
                "node_id": seed_node.id,
                "seed_bootstrap_hash": seed_node.seed_bootstrap_hash,
                "evaluation_binding": binding,
                "values_by_dataset": seed_node.metric.values_by_dataset(),
            }
        )
        for dataset, value in seed_node.metric.values_by_dataset().items():
            values_by_dataset[dataset].append(float(value))
    statistics_by_dataset = {}
    for dataset, values in values_by_dataset.items():
        sample_stdev = statistics.stdev(values)
        standard_error = sample_stdev / math.sqrt(len(values))
        statistics_by_dataset[dataset] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "sample_stdev": sample_stdev,
            "standard_error": standard_error,
            "ci95_half_width": _student_t_95_critical(len(values)) * standard_error,
            "min": min(values),
            "max": max(values),
        }
    attempt = {
        "schema": "xscientist.multi-seed-rejection.v1",
        "stage": stage.name,
        "qualified_node_id": node.id,
        "qualified_code_sha256": "sha256:"
        + hashlib.sha256(node.code.encode("utf-8")).hexdigest(),
        "configured_seeds": list(configured_seeds),
        "evaluation_contract": contract,
        "reason_code": reason_code,
        "seeds": rows,
        "datasets": statistics_by_dataset,
        "stability_policy": {
            "max_relative_ci_half_width": max_relative_ci_half_width,
            "absolute_ci_floor": absolute_ci_floor,
        },
        "control_receipt_hash": (
            control_report.get("receipt_hash")
            if isinstance(control_report, dict)
            else None
        ),
        "required_improvements": required_improvements,
    }
    attempt["receipt_hash"] = _strict_sha256_json(attempt)
    return attempt


def _validate_rejected_multi_seed_attempts(
    journal: Journal,
    node: Node,
    *,
    expected_stage_name: str,
    expected_configured_seeds: list[int],
) -> set[str]:
    expected_code_hash = (
        "sha256:" + hashlib.sha256(node.code.encode("utf-8")).hexdigest()
    )
    member_ids: set[str] = set()
    receipt_hashes: set[str] = set()
    for attempt in node.multi_seed_attempts:
        if not isinstance(attempt, dict):
            raise ExperimentCannotContinueError(
                "Rejected multi-seed receipt is invalid"
            )
        payload = dict(attempt)
        receipt_hash = payload.pop("receipt_hash", None)
        rows = attempt.get("seeds")
        dataset_statistics = attempt.get("datasets")
        configured_seeds = attempt.get("configured_seeds")
        policy = attempt.get("stability_policy")
        control_receipt_hash = attempt.get("control_receipt_hash")
        required_improvements = attempt.get("required_improvements")
        if (
            attempt.get("schema") != "xscientist.multi-seed-rejection.v1"
            or receipt_hash != _strict_sha256_json(payload)
            or receipt_hash in receipt_hashes
            or attempt.get("qualified_node_id") != node.id
            or attempt.get("qualified_code_sha256") != expected_code_hash
            or attempt.get("evaluation_contract") != node.evaluation_comparison_contract
            or attempt.get("reason_code") not in {"stability", "paired_improvement"}
            or attempt.get("stage") != expected_stage_name
            or not isinstance(rows, list)
            or not isinstance(dataset_statistics, dict)
            or not isinstance(configured_seeds, list)
            or configured_seeds != expected_configured_seeds
            or len(rows) != len(configured_seeds)
            or len(rows) < 3
            or [row.get("seed") for row in rows if isinstance(row, dict)]
            != configured_seeds
            or not isinstance(policy, dict)
            or any(
                isinstance(policy.get(key), bool)
                or not isinstance(policy.get(key), (int, float))
                or not math.isfinite(float(policy[key]))
                or float(policy[key]) <= 0
                for key in (
                    "max_relative_ci_half_width",
                    "absolute_ci_floor",
                )
            )
            or (
                control_receipt_hash is not None
                and re.fullmatch(r"sha256:[0-9a-f]{64}", str(control_receipt_hash))
                is None
            )
            or (
                required_improvements is not None
                and (
                    isinstance(required_improvements, bool)
                    or not isinstance(required_improvements, int)
                    or required_improvements < 1
                )
            )
        ):
            raise ExperimentCannotContinueError(
                "Rejected multi-seed receipt is invalid"
            )
        receipt_hashes.add(receipt_hash)
        for row in rows:
            if not isinstance(row, dict):
                raise ExperimentCannotContinueError(
                    "Rejected multi-seed member is invalid"
                )
            seed_node = journal.get_node_by_id(row.get("node_id"))
            expected_code, expected_bootstrap = _inject_seed_bootstrap(
                node.code,
                row.get("seed"),
            )
            if (
                seed_node is None
                or seed_node.id in member_ids
                or seed_node.parent is None
                or seed_node.parent.id != node.id
                or seed_node.is_seed_node is not True
                or seed_node.is_seed_agg_node is True
                or seed_node.random_seed != row.get("seed")
                or seed_node.code != expected_code
                or seed_node.seed_bootstrap_hash != expected_bootstrap
                or row.get("seed_bootstrap_hash") != expected_bootstrap
                or not isinstance(row.get("values_by_dataset"), dict)
                or set(row["values_by_dataset"]) != set(node.metric.values_by_dataset())
                or evaluation_hash_binding(seed_node.evaluation_report)
                != row.get("evaluation_binding")
                or seed_node.evaluation_comparison_contract
                != node.evaluation_comparison_contract
                or seed_node.metric.values_by_dataset() != row.get("values_by_dataset")
            ):
                raise ExperimentCannotContinueError(
                    "Rejected multi-seed member does not resolve to evidence"
                )
            member_ids.add(seed_node.id)
        recomputed_values: dict[str, list[float]] = {
            name: [] for name in node.metric.values_by_dataset()
        }
        for row in rows:
            for dataset, value in row["values_by_dataset"].items():
                recomputed_values[dataset].append(float(value))
        expected_statistics = {}
        for dataset, values in recomputed_values.items():
            sample_stdev = statistics.stdev(values)
            standard_error = sample_stdev / math.sqrt(len(values))
            expected_statistics[dataset] = {
                "n": len(values),
                "mean": statistics.mean(values),
                "sample_stdev": sample_stdev,
                "standard_error": standard_error,
                "ci95_half_width": _student_t_95_critical(len(values)) * standard_error,
                "min": min(values),
                "max": max(values),
            }
        if dataset_statistics != expected_statistics:
            raise ExperimentCannotContinueError(
                "Rejected multi-seed statistics do not match evidence"
            )
    accepted_ids = set()
    if (
        isinstance(node.multi_seed_report, dict)
        and node.multi_seed_report.get("stage") == expected_stage_name
    ):
        accepted_rows = node.multi_seed_report.get("seeds")
        if isinstance(accepted_rows, list):
            accepted_ids = {
                row.get("node_id") for row in accepted_rows if isinstance(row, dict)
            }
    actual_ids = {
        child.id
        for child in node.children
        if child.is_seed_node and not child.is_seed_agg_node
    }
    if actual_ids != member_ids | accepted_ids:
        raise ExperimentCannotContinueError(
            "Seed evidence is not claimed by a finalized or rejected receipt"
        )
    return member_ids


def _build_multi_seed_report(
    *,
    stage: Stage,
    node: Node,
    seed_nodes: list[Node],
    configured_seeds: list[int],
    max_relative_ci_half_width: float,
    absolute_ci_floor: float,
    control_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if (
        len(seed_nodes) != len(configured_seeds)
        or len(seed_nodes) < 3
        or len(set(configured_seeds)) != len(configured_seeds)
        or not math.isfinite(max_relative_ci_half_width)
        or max_relative_ci_half_width <= 0
        or not math.isfinite(absolute_ci_floor)
        or absolute_ci_floor <= 0
    ):
        raise ExperimentCannotContinueError("Multi-seed aggregation inputs are invalid")
    contract = node.evaluation_comparison_contract
    signature = node.metric.comparison_signature
    if contract is None:
        raise ExperimentCannotContinueError(
            "Qualified node lacks comparison-ready evidence"
        )
    member_rows: list[dict[str, Any]] = []
    values_by_dataset: dict[str, list[float]] = {
        name: [] for name in node.metric.values_by_dataset()
    }
    for expected_seed, seed_node in zip(configured_seeds, seed_nodes):
        binding = evaluation_hash_binding(seed_node.evaluation_report)
        values = seed_node.metric.values_by_dataset()
        if (
            seed_node.random_seed != expected_seed
            or not seed_node.has_verified_metric
            or seed_node.evaluation_comparison_contract != contract
            or seed_node.metric.comparison_signature != signature
            or binding is None
            or set(values) != set(values_by_dataset)
        ):
            raise ExperimentCannotContinueError(
                "Seed evidence does not match the qualified comparison contract"
            )
        for dataset, value in values.items():
            values_by_dataset[dataset].append(float(value))
        member_rows.append(
            {
                "seed": expected_seed,
                "node_id": seed_node.id,
                "seed_bootstrap_hash": seed_node.seed_bootstrap_hash,
                "evaluation_binding": binding,
                "values_by_dataset": values,
            }
        )
    statistics_by_dataset: dict[str, dict[str, float | int | bool]] = {}
    for dataset, values in values_by_dataset.items():
        sample_stdev = statistics.stdev(values)
        standard_error = sample_stdev / math.sqrt(len(values))
        half_width = _student_t_95_critical(len(values)) * standard_error
        mean = statistics.mean(values)
        threshold = max(absolute_ci_floor, abs(mean) * max_relative_ci_half_width)
        stable = half_width <= threshold
        statistics_by_dataset[dataset] = {
            "n": len(values),
            "mean": mean,
            "sample_stdev": sample_stdev,
            "standard_error": standard_error,
            "ci95_half_width": half_width,
            "min": min(values),
            "max": max(values),
            "stability_threshold": threshold,
            "stable": stable,
        }
        if not stable:
            raise MultiSeedGateRejectedError("stability")
    report: dict[str, Any] = {
        "schema": "xscientist.multi-seed.v1",
        "stage": stage.name,
        "qualified_node_id": node.id,
        "qualified_code_sha256": "sha256:"
        + hashlib.sha256(node.code.encode("utf-8")).hexdigest(),
        "configured_seeds": list(configured_seeds),
        "evaluation_contract": contract,
        "seeds": member_rows,
        "datasets": statistics_by_dataset,
        "stability_policy": {
            "max_relative_ci_half_width": max_relative_ci_half_width,
            "absolute_ci_floor": absolute_ci_floor,
        },
        "control_receipt_hash": (
            control_report.get("receipt_hash")
            if isinstance(control_report, dict)
            else None
        ),
    }
    report["receipt_hash"] = _strict_sha256_json(report)
    return _validate_multi_seed_report(node, report)


def _multi_seed_improvement_count(
    control_report: dict[str, Any],
    candidate_report: dict[str, Any],
    *,
    maximize: bool,
) -> int:
    if (
        control_report.get("configured_seeds")
        != candidate_report.get("configured_seeds")
        or control_report.get("evaluation_contract")
        != candidate_report.get("evaluation_contract")
        or set(control_report.get("datasets") or {})
        != set(candidate_report.get("datasets") or {})
    ):
        return -1
    control_rows = {
        row["seed"]: row["values_by_dataset"] for row in control_report["seeds"]
    }
    candidate_rows = {
        row["seed"]: row["values_by_dataset"] for row in candidate_report["seeds"]
    }
    improvements = 0
    for name, control in control_report["datasets"].items():
        control_mean = float(control["mean"])
        signed_deltas = []
        for seed in control_report["configured_seeds"]:
            raw_delta = float(candidate_rows[seed][name]) - float(
                control_rows[seed][name]
            )
            signed_deltas.append(raw_delta if maximize else -raw_delta)
        mean_delta = statistics.mean(signed_deltas)
        standard_error = statistics.stdev(signed_deltas) / math.sqrt(len(signed_deltas))
        half_width = _student_t_95_critical(len(signed_deltas)) * standard_error
        tolerance = max(1e-9, abs(control_mean) * 1e-6)
        if mean_delta + half_width < -tolerance:
            return -1
        if mean_delta - half_width > tolerance:
            improvements += 1
    return improvements


class AgentManager:
    def __init__(self, task_desc: str, cfg: Any, workspace_dir: Path):
        _assert_glm53_authority_boundary(cfg)
        self.task_desc = json.loads(task_desc)
        for k in [
            "Title",
            "Abstract",
            "Short Hypothesis",
            "Experiments",
            "Risk Factors and Limitations",
        ]:
            if k not in self.task_desc.keys():
                raise ValueError(f"Key {k} not found in task_desc")
        self.cfg = cfg
        self.workspace_dir = workspace_dir
        self._artifact_root = Path(cfg.log_dir).expanduser().resolve()
        # Capture the immutable scientific input identity once. Repeated stage
        # checkpoints then bind to the run's original inputs without re-hashing
        # large datasets on every save.
        self._input_tree_identity = _input_tree_identity(workspace_dir)
        self.current_stage_number = 0
        self.stages: List[Stage] = []
        self.current_stage: Optional[Stage] = None
        self.journals: Dict[str, Journal] = {}
        self.stage_history: List[StageTransition] = []
        self.completed_stages: List[str] = []
        self.main_stage_dict: Dict[int, str] = dict(MAIN_STAGE_NAMES)
        self.main_stage_goals: Dict[int, str] = {
            1: """
                - Build a basic working control implementation
                - Evaluate the unchanged control on at least THREE identity-bound datasets
                - Save stable sample_ids and exact raw evaluation inputs for every dataset so the host evaluator, not the research Agent, computes record fingerprints
                - Separate the fixed data/split seed from the variable training seed
                - Aim for basic functional correctness with deterministic evidence
                - If you are given \"Code To Use\", you can directly use it as a starting point.""",
            2: """
                - Change hyperparameters such as learning rate, number of epochs, batch size, etc. to improve the performance
                - DO NOT change the model architecture from the previous stage
                - Improve the locked primary metric on the exact same three control datasets without regressing any of them""",
            3: """
                - Explore novel improvements
                - Come up with experiments to reveal new insights
                - Be creative and think outside the box
                - Preserve the exact locked three-dataset evaluation contract""",
            4: """
                - Conduct systematic component analysis that reveals the contribution of each part
                - Use the same datasets you used from the previous stage""",
        }
        # Create initial stage
        self._create_initial_stage()

    def _get_max_iterations(self, stage_number: int) -> int:
        """Get max iterations for a stage from config or default"""
        value = getattr(
            self.cfg.agent.stages,
            f"stage{stage_number}_max_iters",
            self.cfg.agent.steps,
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= MAX_AGENT_STAGE_ITERATIONS
        ):
            raise ValueError("Configured stage iteration budget is invalid")
        return value

    def _get_task_desc_str(self):
        task_desc = """You are an ambitious AI researcher who is looking to publish a paper that will contribute significantly to the field.
You have an idea and you want to conduct creative experiments to gain scientific insights.
Your aim is to run experiments to gather sufficient results for a top conference paper.
Your research idea:\n\n
"""
        task_desc += (
            "Title:\n"
            + self.task_desc["Title"]
            + "\n"
            + "Abstract:\n"
            + self.task_desc["Abstract"]
            + "\n"
            + "Short Hypothesis:\n"
            + self.task_desc["Short Hypothesis"]
            + "\n"
        )
        if "Code" in self.task_desc:
            task_desc += "Code To Use:\n" + self.task_desc["Code"] + "\n"
        research_contract = self.task_desc.get("XScientist Research Contract")
        if isinstance(research_contract, dict):
            task_desc += (
                "\nBinding XScientist Research Contract:\n"
                "This contract is an execution constraint, not optional context. "
                "Run the required tasks and discriminating tests, preserve failed "
                "or refuting outcomes, and report every deviation. Do not claim "
                "completion while an acceptance rule or required artifact is "
                "missing.\n"
                + json.dumps(
                    research_contract,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        return task_desc

    def _create_initial_stage(self):
        """Create the initial stage configuration"""
        self.current_stage_number += 1
        initial_stage = _validate_stage_definition(
            Stage(
                name="1_initial_implementation_1_preliminary",
                description="preliminary",
                goals=self.main_stage_goals[1],
                max_iterations=self._get_max_iterations(self.current_stage_number),
                num_drafts=self.cfg.agent.search.num_drafts,
                stage_number=self.current_stage_number,
            )
        )

        self.stages.append(initial_stage)
        self.current_stage = initial_stage
        self.journals[initial_stage.name] = Journal()

    def _curate_task_desc(self, stage: Stage) -> str:
        task_desc = self._get_task_desc_str()

        if stage.name.startswith("3_"):
            if isinstance(self.task_desc["Experiments"], list):
                if isinstance(self.task_desc["Experiments"][0], str):
                    experiment_str = "\n".join(self.task_desc["Experiments"])
                elif isinstance(self.task_desc["Experiments"][0], dict):
                    experiment_str = "\n".join(
                        [
                            f"{k}: {v}"
                            for d in self.task_desc["Experiments"]
                            for k, v in d.items()
                        ]
                    )
            elif isinstance(self.task_desc["Experiments"], str):
                experiment_str = self.task_desc["Experiments"]
            else:
                raise ValueError(
                    f"Experiments is not a list or string: {self.task_desc['Experiments']}"
                )
            task_desc += "Experiment Plan: " + experiment_str + "\n"
        elif stage.name.startswith("4_"):
            if isinstance(self.task_desc["Risk Factors and Limitations"], list):
                risk_factors_str = "\n".join(
                    self.task_desc["Risk Factors and Limitations"]
                )
            else:
                risk_factors_str = self.task_desc["Risk Factors and Limitations"]
            task_desc += "Risk Factors and Limitations: " + risk_factors_str + "\n"

        return task_desc

    def _save_checkpoint(self):
        """Save the current state of the experiment"""
        checkpoint_stage = self.current_stage or (
            self.stages[-1] if self.stages else None
        )
        if checkpoint_stage is None:
            logger.warning("Cannot save checkpoint: no stages are available")
            return None
        stage_name = "stage_" + checkpoint_stage.name
        save_path = self._artifact_root / stage_name / "checkpoint.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "journals": {
                name: journal.to_dict(artifact_base=self._artifact_root)
                for name, journal in self.journals.items()
            },
            "stages": [stage.__dict__ for stage in self.stages],
            "stage_history": [transition.__dict__ for transition in self.stage_history],
            "completed_stages": self.completed_stages,
            "current_stage_number": self.current_stage_number,
            "task_desc": self.task_desc,
            "input_tree": self._input_tree_identity,
            "current_stage": (
                checkpoint_stage.__dict__ if self.current_stage is not None else None
            ),
        }
        envelope = {
            "schema": CHECKPOINT_SCHEMA,
            "task_fingerprint": _sha256_json(self.task_desc),
            "config_fingerprint": _sha256_json(_checkpoint_config_projection(self.cfg)),
            "payload_hash": _sha256_json(checkpoint),
            "payload": checkpoint,
        }
        print("Saving checkpoint to ", portable_path(save_path, base=Path.cwd()))
        atomic_write_json(
            save_path,
            envelope,
            default=str,
        )
        return save_path

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        cfg: Any,
        workspace_dir: Path,
        expected_task_desc: str | dict | None = None,
    ) -> "AgentManager":
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path.suffix != ".json":
            raise ValueError(
                "Unsafe legacy checkpoint format; JSON checkpoint required"
            )
        try:
            envelope = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid BFTS checkpoint JSON: {exc}") from exc
        checkpoint_schema = envelope.get("schema")
        if checkpoint_schema == "xscientist.bfts.checkpoint.v3":
            raise ValueError(
                "Legacy BFTS checkpoint v3 does not bind input content; "
                "start a new run to create a v5 checkpoint"
            )
        if checkpoint_schema == "xscientist.bfts.checkpoint.v4":
            raise ValueError(
                "Legacy BFTS checkpoint v4 does not bind attempt reservations "
                "or multi-seed stage receipts; start a new run to create a v5 checkpoint"
            )
        if checkpoint_schema != CHECKPOINT_SCHEMA:
            raise ValueError("Unsupported BFTS checkpoint schema")
        checkpoint = envelope.get("payload")
        if not isinstance(checkpoint, dict):
            raise ValueError("Invalid BFTS checkpoint payload")
        if envelope.get("payload_hash") != _sha256_json(checkpoint):
            raise ValueError("BFTS checkpoint content hash mismatch")
        required = {"journals", "task_desc", "input_tree", "current_stage"}
        missing = sorted(required - set(checkpoint))
        if missing:
            raise ValueError(f"BFTS checkpoint is missing fields: {missing}")
        if envelope.get("task_fingerprint") != _sha256_json(checkpoint["task_desc"]):
            raise ValueError("BFTS checkpoint task fingerprint mismatch")
        if expected_task_desc is not None:
            current_task = (
                json.loads(expected_task_desc)
                if isinstance(expected_task_desc, str)
                else expected_task_desc
            )
            if envelope.get("task_fingerprint") != _sha256_json(current_task):
                raise ValueError("BFTS checkpoint does not match the current task")
        if envelope.get("config_fingerprint") != _sha256_json(
            _checkpoint_config_projection(cfg)
        ):
            raise ValueError("BFTS checkpoint configuration fingerprint mismatch")
        checkpoint_input_tree = checkpoint["input_tree"]
        if (
            not isinstance(checkpoint_input_tree, dict)
            or set(checkpoint_input_tree) != {"schema", "fingerprint"}
            or checkpoint_input_tree.get("schema") != INPUT_TREE_SCHEMA
            or not isinstance(checkpoint_input_tree.get("fingerprint"), str)
            or re.fullmatch(
                r"sha256:[0-9a-f]{64}", checkpoint_input_tree["fingerprint"]
            )
            is None
        ):
            raise ValueError("BFTS checkpoint input tree identity is invalid")

        journals_payload = checkpoint["journals"]
        if not isinstance(journals_payload, dict):
            raise ValueError("BFTS checkpoint journals must be an object")
        invalid_journal_names = [
            name
            for name, payload in journals_payload.items()
            if not isinstance(name, str) or not name or not isinstance(payload, dict)
        ]
        if invalid_journal_names:
            raise ValueError("BFTS checkpoint contains invalid journal entries")

        stages_payload = checkpoint.get("stages")
        if not isinstance(stages_payload, list) or not stages_payload:
            raise ValueError("BFTS checkpoint must contain at least one stage")
        stage_names = []
        stage_numbers = []
        for index, stage_payload in enumerate(stages_payload):
            if not isinstance(stage_payload, dict):
                raise ValueError(f"BFTS checkpoint stage {index} must be an object")
            stage_name = stage_payload.get("name")
            if not isinstance(stage_name, str) or not stage_name:
                raise ValueError(f"BFTS checkpoint stage {index} has an invalid name")
            parsed_stage_number, _, _, _ = _parse_stage_name(stage_name)
            stage_number = stage_payload.get("stage_number")
            if (
                isinstance(stage_number, bool)
                or not isinstance(stage_number, int)
                or stage_number < 1
            ):
                raise ValueError(
                    f"BFTS checkpoint stage {stage_name} has an invalid stage number"
                )
            if parsed_stage_number != stage_number:
                raise ValueError(
                    f"BFTS checkpoint stage {stage_name} has a mismatched stage number"
                )
            try:
                _validate_stage_definition(Stage(**stage_payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"BFTS checkpoint stage {stage_name} has an invalid definition"
                ) from exc
            stage_names.append(stage_name)
            stage_numbers.append(stage_number)
        duplicate_stage_names = sorted(
            name for name in set(stage_names) if stage_names.count(name) > 1
        )
        if duplicate_stage_names:
            raise ValueError(
                f"BFTS checkpoint contains duplicate stages: {duplicate_stage_names}"
            )
        if stage_numbers[0] != 1:
            raise ValueError("BFTS checkpoint must begin with main stage 1")
        for index in range(len(stage_numbers) - 1):
            current_number = stage_numbers[index]
            next_number = stage_numbers[index + 1]
            if next_number < current_number or next_number > current_number + 1:
                raise ValueError(
                    "BFTS checkpoint contains an invalid stage number progression"
                )

        stage_history_payload = checkpoint.get("stage_history")
        if not isinstance(stage_history_payload, list):
            raise ValueError("BFTS checkpoint stage history must be a list")
        if len(stage_history_payload) != len(stage_names) - 1:
            raise ValueError(
                "BFTS checkpoint stage history does not match the stage sequence"
            )
        for index, transition_payload in enumerate(stage_history_payload):
            if not isinstance(transition_payload, dict):
                raise ValueError(
                    f"BFTS checkpoint stage transition {index} must be an object"
                )
            if (
                transition_payload.get("from_stage") != stage_names[index]
                or transition_payload.get("to_stage") != stage_names[index + 1]
            ):
                raise ValueError(
                    "BFTS checkpoint stage history does not follow the stage sequence"
                )
            if not isinstance(transition_payload.get("reason"), str) or not isinstance(
                transition_payload.get("config_adjustments"), dict
            ):
                raise ValueError(
                    f"BFTS checkpoint stage transition {index} is malformed"
                )

        stage_name_set = set(stage_names)
        journal_name_set = set(journals_payload)
        missing_journals = sorted(stage_name_set - journal_name_set)
        if missing_journals:
            raise ValueError(
                f"BFTS checkpoint stages are missing journals: {missing_journals}"
            )
        unknown_journals = sorted(journal_name_set - stage_name_set)
        if unknown_journals:
            raise ValueError(
                f"BFTS checkpoint journals reference unknown stages: {unknown_journals}"
            )

        current_stage_payload = checkpoint["current_stage"]
        current_stage_number = checkpoint.get("current_stage_number")
        if (
            isinstance(current_stage_number, bool)
            or not isinstance(current_stage_number, int)
            or current_stage_number < 1
        ):
            raise ValueError("BFTS checkpoint has an invalid current stage number")
        max_stage_number = max(stage_numbers)
        if current_stage_number != max_stage_number:
            raise ValueError(
                "BFTS checkpoint current stage number does not match its stages"
            )
        if current_stage_payload is not None:
            if not isinstance(current_stage_payload, dict):
                raise ValueError("BFTS checkpoint current stage must be an object")
            current_stage_name = current_stage_payload.get("name")
            if current_stage_name not in stage_name_set:
                raise ValueError(
                    "BFTS checkpoint current stage is not present in the stage list"
                )
            canonical_stage_payload = stages_payload[
                stage_names.index(current_stage_name)
            ]
            if current_stage_payload != canonical_stage_payload:
                raise ValueError(
                    "BFTS checkpoint current stage does not match its stage definition"
                )
            if current_stage_payload.get("stage_number") != current_stage_number:
                raise ValueError(
                    "BFTS checkpoint current stage number does not match current stage"
                )
            if current_stage_payload.get("name") != stage_names[-1]:
                raise ValueError(
                    "BFTS checkpoint current stage is not the latest stage"
                )
        elif stage_numbers[-1] != max(MAIN_STAGE_NAMES):
            raise ValueError(
                "A terminal BFTS checkpoint must complete every main stage"
            )

        completed_stages_payload = checkpoint.get("completed_stages")
        if not isinstance(completed_stages_payload, list) or any(
            not isinstance(name, str) or not name for name in completed_stages_payload
        ):
            raise ValueError("BFTS checkpoint completed stages must be a list of names")
        if len(completed_stages_payload) != len(set(completed_stages_payload)):
            raise ValueError("BFTS checkpoint contains duplicate completed stages")
        expected_completed_stages = [
            stage_names[index]
            for index in range(len(stage_names) - 1)
            if stage_numbers[index + 1] > stage_numbers[index]
        ]
        if current_stage_payload is None:
            expected_completed_stages.append(stage_names[-1])
        if completed_stages_payload != expected_completed_stages:
            raise ValueError(
                "BFTS checkpoint completed stages do not match its lifecycle"
            )

        manager = cls(
            task_desc=json.dumps(checkpoint["task_desc"]),
            cfg=cfg,
            workspace_dir=workspace_dir,
        )
        checkpoint_artifact_root = checkpoint_path.parent.parent.resolve()
        manager._artifact_root = checkpoint_artifact_root
        if manager._input_tree_identity != checkpoint_input_tree:
            raise ValueError("BFTS checkpoint input tree fingerprint mismatch")
        manager.journals = {
            name: (
                payload if isinstance(payload, Journal) else Journal.from_dict(payload)
            )
            for name, payload in journals_payload.items()
        }
        _validate_restored_authority_ledger(
            manager.journals,
            log_dir=checkpoint_artifact_root,
        )
        if _is_glm53_stage(cfg.agent.code):
            unsafe_nodes = sorted(
                node.id
                for journal in manager.journals.values()
                for node in journal.nodes
                if isinstance(node.code, str)
                and node.code.strip()
                and not node.is_seed_agg_node
                and (
                    not isinstance(node.implementation_spec, dict)
                    or not isinstance(node.implementation_spec_hash, str)
                )
            )
            if unsafe_nodes:
                raise ValueError(
                    "GLM-5.3 checkpoint contains executable nodes without a "
                    "locked judgment spec; start a new run instead of resuming: "
                    + ", ".join(unsafe_nodes[:10])
                )
        manager.stages = [
            stage if isinstance(stage, Stage) else Stage(**stage)
            for stage in stages_payload
        ]
        manager.stage_history = [
            (
                transition
                if isinstance(transition, StageTransition)
                else StageTransition(**transition)
            )
            for transition in stage_history_payload
        ]
        manager.completed_stages = list(completed_stages_payload)
        manager.current_stage_number = current_stage_number
        current_stage = current_stage_payload
        if isinstance(current_stage, dict):
            current_stage = Stage(**current_stage)
        if current_stage is not None:
            current_stage = next(
                (stage for stage in manager.stages if stage.name == current_stage.name),
                current_stage,
            )
        manager.current_stage = current_stage
        manager.task_desc = checkpoint["task_desc"]
        manager.cfg = cfg
        manager.workspace_dir = workspace_dir
        manager._input_tree_identity = checkpoint_input_tree
        try:
            for stage_index, stage in enumerate(manager.stages):
                ordinary_nodes = [
                    node
                    for node in manager.journals[stage.name].nodes
                    if not node.is_seed_node and not node.is_seed_agg_node
                ]
                inherited_controls = 0
                if stage_index > 0 and ordinary_nodes:
                    previous_stage = manager.stages[stage_index - 1]
                    previous_node = manager.journals[
                        previous_stage.name
                    ].get_node_by_id(previous_stage.qualified_node_id or "")
                    inherited = ordinary_nodes[0]
                    if (
                        previous_node is None
                        or inherited.id != previous_node.id
                        or inherited.code != previous_node.code
                        or inherited.evaluation_report
                        != previous_node.evaluation_report
                        or inherited.multi_seed_report
                        != previous_node.multi_seed_report
                    ):
                        raise ExperimentCannotContinueError(
                            "Checkpoint stage does not begin with its inherited control"
                        )
                    inherited_controls = 1
                if len(ordinary_nodes) - inherited_controls > stage.attempt_count:
                    raise ExperimentCannotContinueError(
                        "Checkpoint journal exceeds its reserved attempt budget"
                    )
            for stage in manager.stages:
                journal = manager.journals[stage.name]
                for node in journal.nodes:
                    manager._replay_node_evaluation(node)
                    _validate_rejected_multi_seed_attempts(
                        journal,
                        node,
                        expected_stage_name=stage.name,
                        expected_configured_seeds=manager._configured_multi_seed_seeds(),
                    )
                    for attempt in node.multi_seed_attempts:
                        manager._validate_rejected_multi_seed_semantics(
                            stage,
                            node,
                            attempt,
                        )
            for stage in manager.stages:
                journal = manager.journals[stage.name]
                local_report_nodes = [
                    node
                    for node in journal.nodes
                    if isinstance(node.multi_seed_report, dict)
                    and node.multi_seed_report.get("stage") == stage.name
                ]
                qualified = (
                    journal.get_node_by_id(stage.qualified_node_id)
                    if stage.qualified_node_id is not None
                    else None
                )
                if stage.qualified_node_id is None:
                    if local_report_nodes:
                        raise ExperimentCannotContinueError(
                            "Unqualified stage node claims a finalized seed receipt"
                        )
                    if stage.multi_seed_receipt_hash is not None:
                        raise ExperimentCannotContinueError(
                            "Stage receipt exists without a qualified node"
                        )
                    if stage.name in manager.completed_stages:
                        raise ExperimentCannotContinueError(
                            "Completed stage lacks a qualified node"
                        )
                    continue
                if (
                    qualified is None
                    or qualified.is_seed_node
                    or qualified.evaluation_comparison_contract is None
                ):
                    raise ExperimentCannotContinueError(
                        "Checkpoint qualified node is invalid"
                    )
                has_report = qualified.multi_seed_report is not None
                has_receipt = stage.multi_seed_receipt_hash is not None
                if has_report != has_receipt:
                    raise ExperimentCannotContinueError(
                        "Checkpoint stage has an incomplete receipt state"
                    )
                if has_report:
                    if [node.id for node in local_report_nodes] != [
                        stage.qualified_node_id
                    ]:
                        raise ExperimentCannotContinueError(
                            "Stage seed receipt is not unique to its qualified node"
                        )
                elif local_report_nodes:
                    raise ExperimentCannotContinueError(
                        "Pending stage lock has an unexpected finalized receipt"
                    )
                complete, _reason = manager._check_stage_completion(stage)
                if not has_receipt and complete:
                    raise ExperimentCannotContinueError(
                        "Pending stage lock unexpectedly passes its final gate"
                    )
                if stage.name in manager.completed_stages and not complete:
                    raise ExperimentCannotContinueError(
                        "Completed checkpoint stage no longer passes its gate"
                    )
                if stage.name in manager.completed_stages and not has_receipt:
                    raise ExperimentCannotContinueError(
                        "Completed checkpoint stage lacks its final receipt"
                    )
                if has_receipt and not complete:
                    raise ExperimentCannotContinueError(
                        "Finalized checkpoint stage no longer passes its gate"
                    )
        except (
            ExperimentCannotContinueError,
            ResearchDecisionError,
            ValueError,
        ) as exc:
            raise ValueError(
                "BFTS checkpoint scientific evidence is inconsistent"
            ) from exc
        return manager

    def _create_agent_for_stage(self, stage: Stage) -> ParallelAgent:
        """Create a ParallelAgent configured for the given stage"""
        stage_cfg = self.cfg.copy()
        stage_cfg.agent.search.num_drafts = stage.num_drafts
        task_desc = self._curate_task_desc(stage)

        (
            main_stage,
            main_stage_name,
            sub_stage_num,
            sub_stage_name,
        ) = self.parse_stage_names(stage.name)
        task_desc = f"{task_desc}\n\nCurrent Main Stage: {main_stage_name}\n"
        task_desc += f"Sub-stage: {sub_stage_num} - {sub_stage_name}\n"
        task_desc += f"Sub-stage goals: {stage.goals}"
        print(f"Preparing research context for stage {stage.name}")

        if main_stage == 2:
            stage1_substages = [s for s in self.stages if s.name.startswith("1_")]
            if not stage1_substages:
                raise ValueError(f"No stage 1 substages found in {self.stages}")
            best_stage1_node = self._get_best_implementation(stage1_substages[-1].name)
            best_stage2_node = None
            best_stage3_node = None
        elif main_stage == 3:
            stage2_substages = [s for s in self.stages if s.name.startswith("2_")]
            if not stage2_substages:
                raise ValueError(f"No stage 2 substages found in {self.stages}")
            best_stage2_node = self._get_best_implementation(stage2_substages[-1].name)
            best_stage1_node = None
            best_stage3_node = None
        elif main_stage == 4:
            # Use the last (sub-)stage's best node
            stage3_substages = [s for s in self.stages if s.name.startswith("3_")]
            if stage3_substages:
                last_substage = stage3_substages[-1]
                best_stage3_node = self._get_best_implementation(last_substage.name)
                best_stage2_node = None
                best_stage1_node = None
            else:
                raise ValueError(f"No stage 3 substages found in {self.stages}")
        else:
            best_stage3_node = None
            best_stage2_node = None
            best_stage1_node = None

        agent = ParallelAgent(
            task_desc=task_desc,
            cfg=stage_cfg,
            journal=self.journals[stage.name],
            stage_name=stage.name,
            best_stage3_node=best_stage3_node,
            best_stage2_node=best_stage2_node,
            best_stage1_node=best_stage1_node,
            evaluation_metric=stage.evaluation_metric,
        )
        if stage.evaluation_metric is None:
            stage.evaluation_metric = agent.evaluation_metrics
            self._save_checkpoint()
        return agent

    def _parse_vlm_feedback(self, node: Node) -> str:
        """Parse the feedback from the VLM"""
        if len(node.plot_analyses) > 0:
            feedback = f"Plot analyses: {node.plot_analyses[0]['analysis']}\n"
        else:
            feedback = "No plot analyses found\n"
            logger.warning(
                f"No plot analyses found for node {node.id} during stage {self.current_stage.name}"
            )
        feedback += f"VLM Feedback Summary: {node.vlm_feedback_summary}\n"
        return feedback

    def _check_substage_completion(
        self, current_substage: Stage, journal: Journal
    ) -> bool:
        """Check if the current sub-stage is complete"""
        best_node = journal.get_best_node_by_metric()
        if best_node is None:
            best_node = journal.get_best_node(cfg=self.cfg)
        if not best_node:
            return False, "No best node found"

        vlm_feedback = self._parse_vlm_feedback(best_node)
        eval_prompt = f"""
        Evaluate if the current sub-stage is complete based on the following evidence:
        1. Figure Analysis:
        {vlm_feedback}

        Requirements for completion:
        - {current_substage.goals}

        Provide a detailed evaluation of completion status.
        """

        try:
            evaluation = query(
                system_message=eval_prompt,
                user_message=None,
                func_spec=stage_completion_eval_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_tokens=_stage_max_tokens(self.cfg.agent.feedback),
            )
            _validate_stage_completion_evaluation(evaluation)
            if evaluation["is_complete"]:
                logger.info("Stage %s completed", current_substage.name)
                print(f"[green]Stage {current_substage.name} completed[/green]")
                return True, "Found working implementation"
            else:
                missing_count = len(evaluation["missing_criteria"])
                logger.info(
                    "Stage %s not complete; missing criteria count=%d",
                    current_substage.name,
                    missing_count,
                )
                print(
                    f"[yellow]Stage {current_substage.name} not complete; "
                    f"missing criteria: {missing_count}[/yellow]"
                )
                return False, f"Missing criteria count: {missing_count}"
        except Exception as e:
            if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                raise
            _fail_research_decision("sub-stage completion evaluation", e)

        # Terminate if max iterations reached
        if len(journal.nodes) >= current_substage.max_iterations:
            logger.info(
                f"Stage {current_substage.name} completed: reached max iterations"
            )
            print(
                f"[green]Stage {current_substage.name} completed: reached max iterations[/green]"
            )
            return True, "Reached max iterations"

        print(f"[green]Stage {current_substage.name} not completed[/green]")
        return False

    def _configured_multi_seed_seeds(self) -> list[int]:
        return _configured_multi_seed_values(self.cfg.agent.multi_seed_eval)

    def _evidence_gate_counts(self) -> tuple[int, int, int]:
        agent_cfg = getattr(self, "cfg", None)
        agent_cfg = getattr(agent_cfg, "agent", None)
        gate_cfg = getattr(agent_cfg, "evidence_gate", None)
        values = (
            getattr(gate_cfg, "minimum_datasets", 3),
            getattr(gate_cfg, "stage2_min_improved", 2),
            getattr(gate_cfg, "stage3_min_improved", 3),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= 64
            for value in values
        ):
            raise ExperimentCannotContinueError(
                "Scientific evidence-gate configuration is invalid"
            )
        minimum, stage2, stage3 = values
        if stage2 > minimum or stage3 > minimum:
            raise ExperimentCannotContinueError(
                "Improvement gates cannot exceed the locked dataset count"
            )
        return minimum, stage2, stage3

    def _replay_node_evaluation(self, node: Node) -> None:
        """Recompute a checkpointed metric from its confined evidence artifact."""

        if not node.has_verified_metric:
            return
        evidence_root = (self._artifact_root / "experiment_results").resolve()
        expected_dir = evidence_root / f"experiment_{node.id}"
        expected_resolved = expected_dir.resolve()
        recorded_dir = Path(str(node.exp_results_dir or ""))
        if not recorded_dir.is_absolute():
            recorded_dir = self._artifact_root / recorded_dir
        if (
            expected_dir.is_symlink()
            or not expected_resolved.is_relative_to(evidence_root)
            or recorded_dir.resolve() != expected_resolved
        ):
            raise ExperimentCannotContinueError(
                "Evaluation evidence directory escapes the run artifact root"
            )
        artifact = expected_dir / "experiment_data.npy"
        code_artifact = expected_dir / "experiment_code.py"
        if (
            artifact.is_symlink()
            or code_artifact.is_symlink()
            or not artifact.is_file()
            or not code_artifact.is_file()
        ):
            raise ExperimentCannotContinueError(
                "Replayable evaluation artifact is missing or unsafe"
            )
        try:
            preserved_code = code_artifact.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ExperimentCannotContinueError(
                "Preserved experiment code cannot be replayed"
            ) from exc
        if preserved_code != node.code:
            raise ExperimentCannotContinueError(
                "Preserved experiment code does not match the checkpoint"
            )
        node.exp_results_dir = str(expected_resolved)
        replay = evaluate_experiment_data(
            artifact,
            requested_metric=node.evaluation_report.get("requested_metric"),
        )
        if (
            replay != node.evaluation_report
            or replay.get("metric") != node.metric.value
        ):
            raise ExperimentCannotContinueError(
                "Replayed evaluation does not match the checkpoint receipt"
            )

    def _validate_report_origin(self, node: Node) -> dict[str, Any]:
        report = _validate_multi_seed_report(node, node.multi_seed_report)
        origin_name = report["stage"]
        origin_stage = next(
            (stage for stage in self.stages if stage.name == origin_name),
            None,
        )
        origin_journal = self.journals.get(origin_name)
        if origin_stage is None or origin_journal is None:
            raise ExperimentCannotContinueError(
                "Multi-seed receipt references an unknown stage"
            )
        origin_node = origin_journal.get_node_by_id(node.id)
        if (
            origin_node is None
            or origin_node.code != node.code
            or origin_node.multi_seed_report != report
        ):
            raise ExperimentCannotContinueError(
                "Multi-seed receipt origin does not match the copied control"
            )
        return self._validate_stage_multi_seed_evidence(
            origin_stage,
            origin_node,
        )

    def _validate_stage_multi_seed_evidence(
        self,
        stage: Stage,
        node: Node,
    ) -> dict[str, Any]:
        journal = self.journals[stage.name]
        report = _validate_multi_seed_journal_links(
            journal,
            node,
            node.multi_seed_report,
        )
        if (
            report["stage"] != stage.name
            or report["configured_seeds"] != self._configured_multi_seed_seeds()
            or stage.multi_seed_receipt_hash != report["receipt_hash"]
        ):
            raise ExperimentCannotContinueError(
                "Stage multi-seed receipt is not bound to its configuration"
            )
        if stage.stage_number == 1:
            if report.get("control_receipt_hash") is not None:
                raise ExperimentCannotContinueError(
                    "Initial-stage receipt must not reference a control"
                )
            return report

        baseline = journal.nodes[0] if journal.nodes else None
        if baseline is None or baseline.id == node.id:
            raise ExperimentCannotContinueError(
                "Stage multi-seed evidence is missing its prior-stage control"
            )
        control_report = self._validate_report_origin(baseline)
        if report.get("control_receipt_hash") != control_report["receipt_hash"]:
            raise ExperimentCannotContinueError(
                "Stage multi-seed receipt is not paired with its control"
            )
        if stage.stage_number in {2, 3}:
            _minimum, stage2_required, stage3_required = self._evidence_gate_counts()
            required = stage2_required if stage.stage_number == 2 else stage3_required
            if (
                _multi_seed_improvement_count(
                    control_report,
                    report,
                    maximize=node.metric._should_maximize(),
                )
                < required
            ):
                raise ExperimentCannotContinueError(
                    "Stage receipt lacks confidence-separated paired improvements"
                )
        elif (
            control_report["configured_seeds"] != report["configured_seeds"]
            or control_report["evaluation_contract"] != report["evaluation_contract"]
        ):
            raise ExperimentCannotContinueError(
                "Ablation receipt is not paired on the control evidence contract"
            )
        return report

    def _validate_rejected_multi_seed_semantics(
        self,
        stage: Stage,
        node: Node,
        attempt: dict[str, Any],
    ) -> None:
        seed_cfg = self.cfg.agent.multi_seed_eval
        expected_policy = {
            "max_relative_ci_half_width": float(
                getattr(seed_cfg, "max_relative_ci_half_width", 0.25)
            ),
            "absolute_ci_floor": float(getattr(seed_cfg, "absolute_ci_floor", 0.01)),
        }
        if attempt["stability_policy"] != expected_policy:
            raise ExperimentCannotContinueError(
                "Rejected seed receipt changed its stability policy"
            )
        unstable = any(
            float(stats["ci95_half_width"])
            > max(
                expected_policy["absolute_ci_floor"],
                abs(float(stats["mean"]))
                * expected_policy["max_relative_ci_half_width"],
            )
            for stats in attempt["datasets"].values()
        )
        control_report = None
        if stage.stage_number > 1:
            journal = self.journals[stage.name]
            if not journal.nodes:
                raise ExperimentCannotContinueError(
                    "Rejected seed receipt lacks its inherited control"
                )
            control_report = self._validate_report_origin(journal.nodes[0])
            if attempt.get("control_receipt_hash") != control_report["receipt_hash"]:
                raise ExperimentCannotContinueError(
                    "Rejected seed receipt changed its paired control"
                )
        elif attempt.get("control_receipt_hash") is not None:
            raise ExperimentCannotContinueError(
                "Initial-stage rejection cannot reference a control"
            )

        reason_code = attempt["reason_code"]
        if reason_code == "stability":
            if not unstable:
                raise ExperimentCannotContinueError(
                    "Stable seed evidence cannot claim a stability rejection"
                )
            return
        if reason_code != "paired_improvement" or stage.stage_number not in {2, 3}:
            raise ExperimentCannotContinueError(
                "Rejected seed receipt reason is inconsistent with its stage"
            )
        if unstable or control_report is None:
            raise ExperimentCannotContinueError(
                "Paired-improvement rejection must first pass stability"
            )
        _, stage2_required, stage3_required = self._evidence_gate_counts()
        required = stage2_required if stage.stage_number == 2 else stage3_required
        if attempt.get("required_improvements") != required:
            raise ExperimentCannotContinueError(
                "Rejected seed receipt changed its improvement gate"
            )
        candidate_report = {
            "configured_seeds": attempt["configured_seeds"],
            "evaluation_contract": attempt["evaluation_contract"],
            "datasets": attempt["datasets"],
            "seeds": attempt["seeds"],
        }
        if (
            _multi_seed_improvement_count(
                control_report,
                candidate_report,
                maximize=node.metric._should_maximize(),
            )
            >= required
        ):
            raise ExperimentCannotContinueError(
                "Improving seed evidence cannot claim paired rejection"
            )

    def _check_stage_completion(self, stage: Stage) -> bool:
        """Apply deterministic scientific gates; exhaustion is never completion."""

        _validate_stage_definition(stage)
        journal = self.journals[stage.name]
        verified = [
            node
            for node in journal.verified_nodes
            if not getattr(node, "is_seed_node", False)
            and not getattr(node, "is_seed_agg_node", False)
        ]
        complete = False
        qualified_node: Node | None = None
        eligible_nodes: list[Node] = []
        reason = "deterministic evidence gate not yet satisfied"
        minimum_datasets, stage2_required, stage3_required = (
            self._evidence_gate_counts()
        )

        if stage.stage_number == 1:
            eligible = [
                node
                for node in verified
                if node.evaluation_comparison_contract is not None
                and not _candidate_was_rejected(node, stage.name)
                and (
                    stage.evaluation_metric is None
                    or node.evaluation_comparison_contract["selected_metric"]
                    == stage.evaluation_metric
                )
                and len(node.evaluation_comparison_contract["datasets"])
                >= minimum_datasets
            ]
            complete = bool(eligible)
            eligible_nodes = eligible
            qualified_node = eligible[0] if eligible else None
            reason = (
                f"Found a verified {minimum_datasets}-dataset control implementation"
                if complete
                else (
                    "No comparison-ready verified control with the configured "
                    "dataset coverage"
                )
            )
        else:
            baseline = journal.nodes[0] if journal.nodes else None
            candidates = [
                node
                for node in verified
                if baseline is not None
                and node.id != baseline.id
                and not _candidate_was_rejected(node, stage.name)
            ]
            if baseline is None or not baseline.has_verified_metric:
                reason = "Verified baseline is missing"
            elif stage.stage_number in {2, 3}:
                required_datasets = (
                    stage2_required if stage.stage_number == 2 else stage3_required
                )
                baseline_contract = baseline.evaluation_comparison_contract
                eligible = [
                    node
                    for node in candidates
                    if baseline_contract is not None
                    and len(baseline_contract["datasets"]) >= required_datasets
                    and _metric_improvement_count(baseline, node) >= required_datasets
                ]
                complete = bool(eligible)
                eligible_nodes = eligible
                qualified_node = (
                    max(
                        eligible,
                        key=lambda node: _qualified_candidate_score(baseline, node),
                    )
                    if eligible
                    else None
                )
                reason = (
                    f"Verified improvement on {required_datasets} locked datasets "
                    "without regression on any locked dataset"
                    if complete
                    else (
                        "No verified candidate both preserves baseline datasets, "
                        f"improves the locked metric, and covers {required_datasets} datasets"
                    )
                )
            else:
                eligible = [
                    node
                    for node in candidates
                    if bool(node.ablation_name)
                    and node.code.strip() != baseline.code.strip()
                    and node.ablation_control_node_id == baseline.id
                    and bool(node.ablation_component)
                    and bool(node.ablation_expected_outcome)
                    and node.ablation_code_diff_hash
                    == _ablation_code_diff_hash(baseline.code, node.code)
                    and node.ablation_control_semantic_hash
                    == _semantic_code_hash(baseline.code)
                    and node.ablation_semantic_hash == _semantic_code_hash(node.code)
                    and node.ablation_semantic_hash
                    != node.ablation_control_semantic_hash
                    and _ablation_component_was_transformed(
                        baseline.code,
                        node.code,
                        node.ablation_component,
                    )
                    and _evaluations_are_comparable(baseline, node)
                ]
                complete = bool(eligible)
                eligible_nodes = eligible
                qualified_node = eligible[0] if eligible else None
                reason = (
                    "Stable paired ablation execution preserved; causal interpretation remains review-gated"
                    if complete
                    else "No deterministically verified comparable ablation result"
                )

        if stage.qualified_node_id:
            locked = next(
                (node for node in eligible_nodes if node.id == stage.qualified_node_id),
                None,
            )
            if locked is None:
                raise ExperimentCannotContinueError(
                    "The locked qualified node no longer satisfies its stage gate"
                )
            if (
                stage.evaluation_metric
                != locked.evaluation_comparison_contract["selected_metric"]
            ):
                raise ExperimentCannotContinueError(
                    "The locked qualified node changed the stage metric"
                )
            qualified_node = locked
            complete = True

        if complete:
            if qualified_node is None:
                raise ExperimentCannotContinueError(
                    "Deterministic gate passed without a qualified node"
                )
            if stage.qualified_node_id is None:
                selected_metric = qualified_node.evaluation_comparison_contract[
                    "selected_metric"
                ]
                if (
                    stage.stage_number > 1
                    and stage.evaluation_metric != selected_metric
                ):
                    raise ExperimentCannotContinueError(
                        "Qualified node changed the locked evaluation metric"
                    )
                stage.evaluation_metric = selected_metric
                stage.qualified_node_id = qualified_node.id
            report = qualified_node.multi_seed_report
            if report is None and stage.multi_seed_receipt_hash is None:
                logger.info(
                    "Stage %s locked candidate %s; multi-seed evidence is pending",
                    stage.name,
                    qualified_node.id,
                )
                return False, "Qualified candidate locked; multi-seed evidence pending"
            if report is None or stage.multi_seed_receipt_hash is None:
                raise ExperimentCannotContinueError(
                    "Stage multi-seed receipt state is incomplete"
                )
            validated_report = self._validate_stage_multi_seed_evidence(
                stage,
                qualified_node,
            )
            if validated_report["receipt_hash"] != stage.multi_seed_receipt_hash:
                raise ExperimentCannotContinueError(
                    "Stage multi-seed receipt does not match its qualified node"
                )
            logger.info("Stage %s passed deterministic evidence gates", stage.name)
            print(f"[green]Stage {stage.name} passed deterministic gates[/green]")
            return True, reason

        attempts = stage.attempt_count
        if attempts >= stage.max_iterations:
            message = (
                f"Stage {stage.name} exhausted its bounded attempts without "
                "satisfying deterministic scientific evidence gates"
            )
            logger.error(message)
            raise ExperimentCannotContinueError(message)

        logger.info(
            "Stage %s remains incomplete after %d/%d attempts",
            stage.name,
            attempts,
            stage.max_iterations,
        )
        return False, reason

    def _get_best_implementation(
        self,
        stage_name: str,
        *,
        require_multi_seed: bool = True,
    ) -> Optional[Node]:
        """Copy the exact node selected by the deterministic stage gate."""
        if stage_name not in self.journals:
            return None
        stage = next((item for item in self.stages if item.name == stage_name), None)
        if stage is None or not stage.qualified_node_id:
            return None
        best_node = self.journals[stage_name].get_node_by_id(stage.qualified_node_id)
        if best_node is not None and not best_node.has_verified_metric:
            raise ExperimentCannotContinueError(
                "Qualified stage node no longer has verified evidence"
            )
        if best_node is not None and require_multi_seed:
            if (
                stage.multi_seed_receipt_hash is None
                or best_node.multi_seed_report is None
                or self._validate_stage_multi_seed_evidence(
                    stage,
                    best_node,
                )["receipt_hash"]
                != stage.multi_seed_receipt_hash
            ):
                raise ExperimentCannotContinueError(
                    "Qualified stage node lacks finalized multi-seed evidence"
                )
        if best_node:
            # Create a clean copy of the node for the next stage
            copied_node = copy.deepcopy(best_node)
            # Reset parent relationship and children
            copied_node.parent = None
            copied_node.children = set()
            return copied_node
        return None

    def _generate_substage_goal(self, main_stage_goal: str, journal: Journal) -> str:
        """Generate the next sub-stage goal based on what has been done so far.

        Args:
            main_stage_goal: The overall goal for the current main stage
            journal: Journal containing the results and progress so far

        Returns:
            str: Specific goals for the next sub-stage
        """
        # Gather current progress metrics
        metrics = self._gather_stage_metrics(journal)
        issues = self._identify_issues(journal)
        progress = self._analyze_progress(journal)

        # Create prompt for the LLM
        prompt = f"""
        Based on the current experimental progress, generate focused goals for the next sub-stage.

        Main Stage Goals:
        {main_stage_goal}

        Current Progress:
        - Total attempts: {metrics['total_nodes']}
        - Successful implementations: {metrics['good_nodes']}
        - Best performance: {metrics['best_metric']['value'] if metrics['best_metric'] else 'N/A'}
        - Convergence status: {progress['convergence_status']}

        Current Issues:
        {json.dumps(issues, indent=2)}

        Recent Changes:
        {json.dumps(progress['recent_changes'], indent=2)}

        Generate specific, actionable sub-stage goals that:
        1. Address current issues and limitations
        2. Build on recent progress
        3. Move towards main stage goals
        4. Are concrete and measurable
        """

        # Define the function specification for the LLM
        substage_goal_spec = FunctionSpec(
            name="generate_substage_goals",
            description="Generate specific goals for the next experimental sub-stage",
            json_schema={
                "type": "object",
                "properties": {
                    "goals": {
                        "type": "string",
                        "description": "Detailed, specific goals for the next sub-stage",
                    },
                    "sub_stage_name": {
                        "type": "string",
                        "pattern": STAGE_NAME_COMPONENT_PATTERN,
                        "description": "The name of the next sub-stage",
                    },
                },
                "required": ["goals", "sub_stage_name"],
                "additionalProperties": False,
            },
        )

        try:
            # Get response from LLM
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=substage_goal_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_tokens=_stage_max_tokens(self.cfg.agent.feedback),
            )

            # Format the response into a structured goal string
            goal_str = f"""
            {response['goals']}
            """
            sub_stage_name = response["sub_stage_name"]
            if not STAGE_NAME_COMPONENT_RE.fullmatch(sub_stage_name):
                raise FunctionCallValidationError("Sub-stage name is invalid")

            return goal_str.strip(), sub_stage_name

        except Exception as e:
            if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                raise
            _fail_research_decision("sub-stage goal generation", e)

    def _create_next_substage(
        self, current_substage: Stage, journal: Journal, substage_feedback: str
    ) -> Optional[Stage]:
        """Create the next sub-stage. Ask LLM to come up with the next sub-stage name and goals
        based on what has been done so far.
        """
        main_stage_num, main_stage_name, sub_stage_num, _ = self.parse_stage_names(
            current_substage.name
        )
        main_stage_goal = self.main_stage_goals[main_stage_num]
        sub_stage_goal, sub_stage_name = self._generate_substage_goal(
            main_stage_goal, journal
        )

        return _validate_stage_definition(
            Stage(
                name=f"{main_stage_num}_{main_stage_name}_{sub_stage_num + 1}_{sub_stage_name}",
                description=sub_stage_name,
                goals="Main stage goals:\n"
                + main_stage_goal
                + "\n\nSub-stage goals:\n"
                + sub_stage_goal,
                max_iterations=self._get_max_iterations(main_stage_num),
                num_drafts=0,
                # `stage_number` is the MAIN stage id (1..4). Keep it stable across substages.
                stage_number=main_stage_num,
            )
        )

    def _create_next_main_stage(
        self, current_substage: Stage, journal: Journal
    ) -> Optional[Stage]:
        (
            main_stage_num,
            main_stage_name,
            sub_stage_num,
            sub_stage_name,
        ) = self.parse_stage_names(current_substage.name)
        if main_stage_num == 4:
            return None
        next_main_stage_name = self.main_stage_dict[main_stage_num + 1]
        sub_stage_num = 1
        sub_stage_name = "first_attempt"
        num_drafts = 0
        stage_number = main_stage_num + 1
        description = f"first_attempt"
        main_stage_goal = self.main_stage_goals[main_stage_num + 1]
        qualified = journal.get_node_by_id(current_substage.qualified_node_id or "")
        if qualified is None or qualified.evaluation_comparison_contract is None:
            raise ExperimentCannotContinueError(
                "Next stage cannot inherit an unbound evaluation metric"
            )
        evaluation_metric = qualified.evaluation_comparison_contract["selected_metric"]

        return _validate_stage_definition(
            Stage(
                name=f"{main_stage_num + 1}_{next_main_stage_name}_{sub_stage_num}_{sub_stage_name}",
                description=description,
                goals=main_stage_goal,
                max_iterations=self._get_max_iterations(main_stage_num + 1),
                num_drafts=num_drafts,
                stage_number=stage_number,
                evaluation_metric=evaluation_metric,
            )
        )

    def _advance_main_stage(self) -> Optional[Stage]:
        """Commit a completed main-stage transition before checkpointing it."""

        completed_stage = self.current_stage
        if completed_stage is None:
            self._save_checkpoint()
            return None

        stage_complete, _reason = self._check_stage_completion(completed_stage)
        if not stage_complete:
            raise ExperimentCannotContinueError(
                "Stage transition requires a valid multi-seed evidence receipt"
            )

        next_main_stage = self._create_next_main_stage(
            completed_stage,
            self.journals[completed_stage.name],
        )
        snapshot = {
            "completed_stages": list(self.completed_stages),
            "stage_history": list(self.stage_history),
            "stages": list(self.stages),
            "journals": dict(self.journals),
            "current_stage": self.current_stage,
            "current_stage_number": self.current_stage_number,
        }
        try:
            if completed_stage.name not in self.completed_stages:
                self.completed_stages.append(completed_stage.name)

            if next_main_stage is not None:
                self.stage_history.append(
                    StageTransition(
                        from_stage=completed_stage.name,
                        to_stage=next_main_stage.name,
                        reason=f"Moving to {next_main_stage.description}",
                        config_adjustments={},
                    )
                )
                self.stages.append(next_main_stage)
                self.journals[next_main_stage.name] = Journal()
                self.current_stage = next_main_stage
                self.current_stage_number = next_main_stage.stage_number
            else:
                logger.info(f"Completed stage: {completed_stage.name}")
                logger.info("No more stages to run -- exiting the loop...")
                self.current_stage = None

            # Persist the state after the transition so resume starts from the next
            # stage (or knows the run is terminal) instead of repeating completed work.
            self._save_checkpoint()
        except BaseException:
            self.completed_stages = snapshot["completed_stages"]
            self.stage_history = snapshot["stage_history"]
            self.stages = snapshot["stages"]
            self.journals = snapshot["journals"]
            self.current_stage = snapshot["current_stage"]
            self.current_stage_number = snapshot["current_stage_number"]
            raise
        return next_main_stage

    def _finalize_multi_seed_gate(
        self,
        stage: Stage,
        qualified_copy: Node,
        seed_nodes: list[Node],
    ) -> dict[str, Any]:
        journal = self.journals[stage.name]
        qualified = journal.get_node_by_id(stage.qualified_node_id or "")
        if qualified is None or qualified.id != qualified_copy.id:
            raise ExperimentCannotContinueError(
                "Multi-seed evaluation is not bound to the qualified node"
            )
        seed_cfg = self.cfg.agent.multi_seed_eval
        configured_seeds = self._configured_multi_seed_seeds()
        relative_limit = getattr(seed_cfg, "max_relative_ci_half_width", 0.25)
        absolute_floor = getattr(seed_cfg, "absolute_ci_floor", 0.01)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in (relative_limit, absolute_floor)
        ):
            raise ExperimentCannotContinueError(
                "Multi-seed stability configuration is invalid"
            )
        control = journal.nodes[0] if stage.stage_number > 1 and journal.nodes else None
        control_report = None
        if control is not None:
            control_report = self._validate_report_origin(control)
        report = _build_multi_seed_report(
            stage=stage,
            node=qualified,
            seed_nodes=seed_nodes,
            configured_seeds=configured_seeds,
            max_relative_ci_half_width=float(relative_limit),
            absolute_ci_floor=float(absolute_floor),
            control_report=control_report,
        )
        if control_report is not None:
            if stage.stage_number in {2, 3}:
                _minimum, stage2_required, stage3_required = (
                    self._evidence_gate_counts()
                )
                required = (
                    stage2_required if stage.stage_number == 2 else stage3_required
                )
                improvement_count = _multi_seed_improvement_count(
                    control_report,
                    report,
                    maximize=qualified.metric._should_maximize(),
                )
                if improvement_count < required:
                    raise MultiSeedGateRejectedError("paired_improvement")
            elif control_report.get("configured_seeds") != report.get(
                "configured_seeds"
            ) or control_report.get("evaluation_contract") != report.get(
                "evaluation_contract"
            ):
                raise ExperimentCannotContinueError(
                    "Ablation evidence is not paired with its control receipt"
                )
        original_report = copy.deepcopy(qualified.multi_seed_report)
        original_receipt = stage.multi_seed_receipt_hash
        try:
            qualified.multi_seed_report = report
            stage.multi_seed_receipt_hash = report["receipt_hash"]
            return self._validate_stage_multi_seed_evidence(stage, qualified)
        except BaseException:
            qualified.multi_seed_report = original_report
            stage.multi_seed_receipt_hash = original_receipt
            raise

    def run(self, exec_callback, step_callback=None):
        """Run the experiment through generated stages"""
        while self.current_stage:  # Main stage loop
            main_stage = self.parse_stage_names(self.current_stage.name)[0]
            print(f"[green]Starting main stage: {main_stage}[/green]")
            print("[cyan]Research goals loaded from the stage contract[/cyan]")

            current_substage = self.current_stage
            while current_substage:  # Sub-stage loop
                # Keep `self.current_stage` aligned with the active (sub-)stage so
                # checkpoints and summaries reflect the actual current state.
                self.current_stage = current_substage
                print(f"[green]Starting sub-stage: {current_substage.name}[/green]")

                with self._create_agent_for_stage(current_substage) as agent:
                    # Initialize with best result from previous sub-stage if available
                    if (
                        self.stage_history
                        and not self.journals[current_substage.name].nodes
                    ):
                        prev_stage = self.stage_history[-1].from_stage
                        print("[cyan]Loading the verified prior-stage baseline[/cyan]")
                        prev_best = self._get_best_implementation(prev_stage)
                        if prev_best:
                            self.journals[self.current_stage.name].append(prev_best)
                        else:
                            raise ExperimentCannotContinueError(
                                "No previous best implementation found for "
                                f"{self.current_stage.name} from stage {prev_stage}"
                            )

                    # Run until sub-stage completion
                    while True:
                        (
                            main_stage_complete,
                            main_stage_feedback,
                        ) = self._check_stage_completion(current_substage)
                        print(
                            f"[cyan]Feedback from _check_stage_completion: {main_stage_feedback}[/cyan]"
                        )

                        # A qualified candidate is a durable intermediate state.
                        # It must receive its complete multi-seed receipt before
                        # any more research attempts are reserved or submitted.
                        if (
                            not main_stage_complete
                            and current_substage.qualified_node_id is not None
                        ):
                            qualified = self._get_best_implementation(
                                current_substage.name,
                                require_multi_seed=False,
                            )
                            if qualified is None:
                                raise ExperimentCannotContinueError(
                                    "Locked stage candidate is missing"
                                )
                            # Candidate identity and metric are the durable
                            # commit marker for the upcoming seed transaction.
                            self._save_checkpoint()
                            journal = self.journals[current_substage.name]
                            journal_size = len(journal.nodes)
                            original_report = copy.deepcopy(
                                journal.get_node_by_id(qualified.id).multi_seed_report
                            )
                            original_attempts = copy.deepcopy(
                                journal.get_node_by_id(qualified.id).multi_seed_attempts
                            )
                            original_receipt = current_substage.multi_seed_receipt_hash
                            try:
                                seed_nodes = agent._run_multi_seed_evaluation(qualified)
                                self._finalize_multi_seed_gate(
                                    current_substage,
                                    qualified,
                                    seed_nodes,
                                )
                                self._save_checkpoint()
                            except MultiSeedGateRejectedError as exc:
                                locked = journal.get_node_by_id(qualified.id)
                                if locked is None:
                                    raise ExperimentCannotContinueError(
                                        "Rejected candidate disappeared from its journal"
                                    ) from None
                                try:
                                    seed_cfg = self.cfg.agent.multi_seed_eval
                                    relative_limit = float(
                                        getattr(
                                            seed_cfg,
                                            "max_relative_ci_half_width",
                                            0.25,
                                        )
                                    )
                                    absolute_floor = float(
                                        getattr(seed_cfg, "absolute_ci_floor", 0.01)
                                    )
                                    control_report = None
                                    required_improvements = None
                                    if current_substage.stage_number > 1:
                                        control_report = self._validate_report_origin(
                                            journal.nodes[0]
                                        )
                                    if current_substage.stage_number in {2, 3}:
                                        _, stage2_required, stage3_required = (
                                            self._evidence_gate_counts()
                                        )
                                        required_improvements = (
                                            stage2_required
                                            if current_substage.stage_number == 2
                                            else stage3_required
                                        )
                                    attempt = _build_rejected_multi_seed_attempt(
                                        stage=current_substage,
                                        node=locked,
                                        seed_nodes=seed_nodes,
                                        configured_seeds=self._configured_multi_seed_seeds(),
                                        reason_code=exc.reason_code,
                                        max_relative_ci_half_width=relative_limit,
                                        absolute_ci_floor=absolute_floor,
                                        control_report=control_report,
                                        required_improvements=required_improvements,
                                    )
                                    locked.multi_seed_report = original_report
                                    locked.multi_seed_attempts.append(attempt)
                                    self._validate_rejected_multi_seed_semantics(
                                        current_substage,
                                        locked,
                                        attempt,
                                    )
                                    current_substage.multi_seed_receipt_hash = (
                                        original_receipt
                                    )
                                    current_substage.qualified_node_id = None
                                    self._save_checkpoint()
                                except BaseException:
                                    locked.multi_seed_report = original_report
                                    locked.multi_seed_attempts = original_attempts
                                    locked.children = {
                                        child
                                        for child in locked.children
                                        if child in journal.nodes[:journal_size]
                                    }
                                    del journal.nodes[journal_size:]
                                    current_substage.multi_seed_receipt_hash = (
                                        original_receipt
                                    )
                                    current_substage.qualified_node_id = qualified.id
                                    raise
                                if step_callback:
                                    step_callback(current_substage, journal)
                                logger.info(
                                    "Preserved rejected multi-seed evidence for node %s (%s)",
                                    locked.id,
                                    exc.reason_code,
                                )
                                continue
                            except BaseException:
                                locked = journal.get_node_by_id(qualified.id)
                                if locked is not None:
                                    locked.multi_seed_report = original_report
                                    locked.children = {
                                        child
                                        for child in locked.children
                                        if child in journal.nodes[:journal_size]
                                    }
                                del journal.nodes[journal_size:]
                                current_substage.multi_seed_receipt_hash = (
                                    original_receipt
                                )
                                raise
                            if step_callback:
                                step_callback(current_substage, journal)

                            # Plot aggregation is explanatory, not progression
                            # authority. Scientific evidence is checkpointed first.
                            try:
                                agent._run_plot_aggregation(qualified, seed_nodes)
                                self._save_checkpoint()
                                if step_callback:
                                    step_callback(current_substage, journal)
                            except Exception as exc:
                                if isinstance(
                                    exc, ResearchDecisionError
                                ) or is_llm_budget_exception(exc):
                                    raise
                                logger.warning(
                                    "Optional multi-seed plot aggregation failed: %s",
                                    type(exc).__name__,
                                )

                            (
                                main_stage_complete,
                                main_stage_feedback,
                            ) = self._check_stage_completion(current_substage)

                        if main_stage_complete:
                            journal = self.journals[current_substage.name]
                            qualified = self._get_best_implementation(
                                current_substage.name
                            )
                            if qualified is None:
                                raise ExperimentCannotContinueError(
                                    "Completed stage lacks its qualified implementation"
                                )
                            current_results = {
                                "metrics": self._gather_stage_metrics(
                                    Journal(nodes=[qualified])
                                ),
                                "issues": self._identify_issues(journal),
                                "progress": self._analyze_progress(journal),
                            }
                            evaluation = {
                                "ready_for_next_stage": True,
                                "reasoning": (
                                    "artifact-internal deterministic gate and "
                                    "bound multi-seed receipt passed; external "
                                    "ground-truth validity remains a review item"
                                ),
                                "recommendations": [],
                                "suggested_focus": None,
                            }
                            self._save_stage_summary(current_results, evaluation)

                            # Exit the loop to move to next main stage
                            current_substage = None
                            break

                        remaining = (
                            current_substage.max_iterations
                            - current_substage.attempt_count
                        )
                        if remaining <= 0:
                            self._check_stage_completion(current_substage)
                            raise ExperimentCannotContinueError(
                                "Stage attempt budget was exhausted"
                            )
                        batch_size = min(agent.num_workers, remaining)
                        # Reserve the exact experiment attempts before worker
                        # submission so a crash cannot replenish the budget.
                        current_substage.attempt_count += batch_size
                        self._save_checkpoint()
                        try:
                            submitted = agent.step(
                                exec_callback,
                                max_new_nodes=batch_size,
                            )
                            if submitted != batch_size:
                                raise ExperimentCannotContinueError(
                                    "Research worker batch did not honor its reserved attempts"
                                )
                        finally:
                            self._save_checkpoint()
                        if step_callback:
                            step_callback(
                                current_substage,
                                self.journals[current_substage.name],
                            )
            # Main stage complete - create next main stage
            if self.current_stage:
                self._advance_main_stage()
            else:
                self._save_checkpoint()

    def _create_stage_analysis_prompt(
        self,
        previous_stages: List[Stage],
        previous_results: Optional[Dict[str, Any]],
        is_initial_stage: bool,
    ) -> str:
        """Create detailed prompt to determine next stage configuration"""
        prompt_parts = [
            f"Task Description: {self._curate_task_desc(previous_stages[-1])}",
            f"Current Stage Number: {previous_stages[-1].stage_number}",
        ]

        if previous_stages:
            stage_history = "\n".join(
                f"Stage {i+1}: {stage.name} - {stage.description}"
                for i, stage in enumerate(previous_stages)
            )
            prompt_parts.append(f"Previous Stages:\n{stage_history}")

        if previous_results:
            # Format node summaries
            if "node_summaries" in previous_results["metrics"]:
                summaries = "\n".join(
                    f"Node {i}: {summary}"
                    for i, summary in enumerate(
                        previous_results["metrics"]["node_summaries"]
                    )
                )
                prompt_parts.append(f"Node Analysis:\n{summaries}")

            # Format VLM feedback and plot analysis
            if "plot_insights" in previous_results:
                plot_insights = previous_results["plot_insights"]
                prompt_parts.append("Visual Analysis Findings:")
                for analysis in plot_insights["analyses"]:
                    prompt_parts.append(f"- {analysis['analysis']}")

            # Format other metrics and findings
            metrics_summary = (
                f"Progress Summary:\n"
                f"- Total attempts: {previous_results['metrics']['total_nodes']}\n"
                f"- Successful implementations: {previous_results['metrics']['good_nodes']}\n"
                f"- Failed attempts: {previous_results['metrics']['buggy_nodes']}\n"
                f"- Best performance: {previous_results['metrics']['best_metric']['value'] if previous_results['metrics']['best_metric'] else 'N/A'}\n"
                f"- Issues identified: {', '.join(previous_results['issues'])}\n"
                f"- Progress status: {previous_results['progress']['convergence_status']}"
            )
            prompt_parts.append(metrics_summary)

            # Save stage transition analysis to notes directory
            base_dir = Path(self.workspace_dir).parent.parent
            run_name = Path(self.workspace_dir).name
            notes_dir = (
                base_dir
                / "logs"
                / run_name
                / "notes"
                / f"stage_{stage_number-1}_to_{stage_number}"
            )
            notes_dir.mkdir(parents=True, exist_ok=True)

            analysis_data = {
                "stage_transition": {
                    "from_stage": stage_number - 1,
                    "to_stage": stage_number,
                    "is_initial_stage": is_initial_stage,  # Add flag for initial stage
                    "metrics_summary": metrics_summary,
                    "node_summaries": previous_results["metrics"].get(
                        "node_summaries", []
                    ),
                    "plot_insights": previous_results.get("plot_insights", {}),
                    "issues": previous_results["issues"],
                    "progress": previous_results["progress"],
                }
            }

            atomic_write_json(
                notes_dir / "stage_transition_analysis.json",
                analysis_data,
            )

        prompt_parts.append(
            "Based on the above comprehensive analysis, determine the appropriate "
            "configuration for the next experimental stage. Consider:\n"
            "1. Visual analysis insights from plots\n"
            "2. Individual node performance and patterns\n"
            "3. Overall progress and convergence status\n"
            "4. Identified issues and challenges\n\n"
            "Include:\n"
            "1. Stage name (brief, descriptive)\n"
            "2. Detailed description of the stage's purpose\n"
            "3. Specific, measurable goals\n"
            "4. Maximum iterations needed\n"
            "5. Success metric threshold (if applicable)"
        )

        return "\n\n".join(prompt_parts)

    def parse_stage_names(self, stage_name: str) -> Tuple[int, str, int, str]:
        """Parse stage name into main stage number, main stage name,
        sub-stage number, and sub-stage name"""
        return _parse_stage_name(stage_name)

    def _save_stage_summary(
        self, current_results: Dict[str, Any], evaluation: Dict[str, Any]
    ):
        """Save comprehensive stage completion summary"""
        base_dir = Path(self.workspace_dir).parent.parent
        run_name = Path(self.workspace_dir).name
        notes_dir = (
            base_dir
            / "logs"
            / run_name
            / "notes"
            / f"stage_{self.current_stage.stage_number}_complete"
        )
        notes_dir.mkdir(parents=True, exist_ok=True)

        completion_data = {
            "stage_completion": {
                "stage_number": self.current_stage.stage_number,
                "stage_name": self.current_stage.name,
                "final_metrics": current_results["metrics"],
                "identified_issues": current_results["issues"],
                "progress_analysis": current_results["progress"],
                "plot_insights": current_results.get("plot_insights", {}),
                "progression_evaluation": {
                    "authority": "deterministic_scientific_gate",
                    "research_agent_self_approval_used": False,
                    "ready_for_next_stage": evaluation["ready_for_next_stage"],
                    "reasoning": evaluation["reasoning"],
                    "recommendations": evaluation["recommendations"],
                    "suggested_focus": evaluation.get("suggested_focus"),
                },
            }
        }

        atomic_write_json(
            notes_dir / "stage_completion_summary.json",
            completion_data,
        )

    def _get_response(self, prompt: str) -> Dict[str, Any]:
        """Get structured response from LLM for stage configuration.

        Args:
            prompt: The analysis prompt to send to the LLM

        Returns:
            Dictionary containing stage configuration with keys:
            - name: str
            - description: str
            - goals: List[str]
            - max_iterations: int
            - success_metric_threshold: Optional[float]
        """
        try:
            response = query(
                system_message=prompt,
                user_message=None,
                func_spec=stage_config_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_tokens=_stage_max_tokens(self.cfg.agent.feedback),
            )
            if not STAGE_NAME_COMPONENT_RE.fullmatch(response["name"]):
                raise FunctionCallValidationError("Stage name is invalid")
            if not 1 <= response["max_iterations"] <= MAX_AGENT_STAGE_ITERATIONS:
                raise FunctionCallValidationError("Stage iteration budget is invalid")
            return response

        except Exception as e:
            if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                raise
            _fail_research_decision("stage configuration", e)

    def _gather_stage_metrics(self, journal: Journal) -> Dict[str, Any]:
        """Gather detailed metrics and analysis from the stage's nodes"""
        metrics = {
            "total_nodes": len(journal.nodes),
            "good_nodes": len(journal.verified_nodes),
            "runnable_nodes": len(journal.good_nodes),
            "advisory_metric_nodes": sum(
                node.metric_provenance == "agent_reported_advisory"
                for node in journal.nodes
            ),
            "buggy_nodes": len(journal.buggy_nodes),
            "best_metric": None,
            "node_summaries": [],
            "vlm_feedback": [],
        }

        # Gather individual node summaries
        for node in journal.nodes:
            if hasattr(node, "_agent"):
                node_summary = node._agent._generate_node_summary(node)
                metrics["node_summaries"].append(node_summary)

        # Get VLM feedback from plot analysis
        for node in journal.verified_nodes:
            if hasattr(node, "_vlm_feedback"):
                metrics["vlm_feedback"].append(node._vlm_feedback)

        best_node = journal.get_best_node_by_metric()
        if best_node:
            best_mean = None
            best_objective = None
            try:
                best_mean = float(best_node.metric.get_mean_value())
                best_objective = (
                    best_mean if best_node.metric._should_maximize() else -best_mean
                )
            except Exception:
                best_mean = None
                best_objective = None

            seed_values = []
            for node in journal.nodes:
                if not getattr(node, "is_seed_node", False):
                    continue
                if node.parent is None or node.parent.id != best_node.id:
                    continue
                if getattr(node, "is_buggy", False):
                    continue
                if node.metric_provenance != "deterministic_verified":
                    continue
                try:
                    seed_values.append(float(node.metric.get_mean_value()))
                except Exception:
                    continue

            seed_eval = None
            if seed_values:
                import statistics

                seed_eval = {
                    "count": len(seed_values),
                    "mean": statistics.mean(seed_values),
                    "stdev": (
                        statistics.pstdev(seed_values) if len(seed_values) > 1 else 0.0
                    ),
                }

            metrics["best_metric"] = {
                "node_id": best_node.id,
                "value": best_node.metric.value,
                "mean": best_mean,
                "objective": best_objective,
                "dataset_names": [
                    ds for ds in (best_node.datasets_successfully_tested or []) if ds
                ],
                "seed_eval": seed_eval,
                "name": (
                    best_node.metric.name
                    if hasattr(best_node.metric, "name")
                    else "validation_metric"
                ),
                "maximize": (
                    best_node.metric.maximize
                    if hasattr(best_node.metric, "maximize")
                    else None
                ),
                "analysis": (
                    best_node.analysis if hasattr(best_node, "analysis") else None
                ),
            }

        return metrics

    def qualified_report_journals(self) -> list[tuple[str, Journal]]:
        """Return one gate-qualified, receipt-bound journal per main stage.

        Final scientific reports must not select from the exploratory journal.
        This view contains only the stage's qualified implementation and the
        exact confirmation-seed nodes claimed by its validated receipt.
        """

        report_stages: dict[int, tuple[Stage, Journal, Node, dict[str, Any]]] = {}
        completed = set(self.completed_stages)
        for stage in self.stages:
            if stage.name not in completed:
                continue
            journal = self.journals.get(stage.name)
            qualified = (
                journal.get_node_by_id(stage.qualified_node_id)
                if journal is not None and stage.qualified_node_id is not None
                else None
            )
            if journal is None or qualified is None:
                raise ExperimentCannotContinueError(
                    "Completed stage lacks its qualified report evidence"
                )
            report = self._validate_report_origin(qualified)
            if (
                report.get("stage") != stage.name
                or report.get("receipt_hash") != stage.multi_seed_receipt_hash
            ):
                raise ExperimentCannotContinueError(
                    "Completed stage report receipt does not match its gate"
                )
            if stage.stage_number in report_stages:
                raise ExperimentCannotContinueError(
                    "Multiple completed substages claim the same main report stage"
                )
            report_stages[stage.stage_number] = (stage, journal, qualified, report)

        expected_stages = set(range(1, len(self.main_stage_dict) + 1))
        if set(report_stages) != expected_stages:
            raise ExperimentCannotContinueError(
                "Final report requires one completed qualified receipt per main stage"
            )

        result: list[tuple[str, Journal]] = []
        for stage_number in sorted(report_stages):
            stage, journal, qualified, report = report_stages[stage_number]
            qualified_copy = copy.deepcopy(qualified)
            qualified_copy.parent = None
            qualified_copy.children = set()
            report_nodes = [qualified_copy]
            for row in report["seeds"]:
                seed = journal.get_node_by_id(row["node_id"])
                if seed is None:
                    raise ExperimentCannotContinueError(
                        "Final report receipt references a missing seed node"
                    )
                seed_copy = copy.deepcopy(seed)
                seed_copy.parent = qualified_copy
                seed_copy.children = set()
                qualified_copy.children.add(seed_copy)
                report_nodes.append(seed_copy)
            result.append((stage.name, Journal(nodes=report_nodes)))
        return result

    def _identify_issues(self, journal: Journal) -> List[str]:
        """Identify systemic issues and challenges from the current stage's results"""
        issues = []

        # Look for patterns in leaf nodes (endpoints of improvement attempts)
        leaf_nodes = [n for n in journal.nodes if n.is_leaf]
        buggy_leaves = [n for n in leaf_nodes if n.is_buggy]

        # If we have buggy leaf nodes, it means we couldn't fix some issues
        if buggy_leaves:
            # Group similar issues
            error_patterns = {}
            for node in buggy_leaves:
                if hasattr(node, "analysis"):
                    # Use the error message as key to group similar issues
                    error_patterns.setdefault(node.analysis, []).append(node.id)

            # Report persistent issues
            for error_msg, node_ids in error_patterns.items():
                if len(node_ids) >= 2:  # If same error occurs multiple times
                    issues.append(f"Persistent issue in nodes {node_ids}: {error_msg}")

        # Include VLM-identified systemic issues
        vlm_issues = set()  # Use set to avoid duplicate issues
        for node in journal.verified_nodes:
            if hasattr(node, "_vlm_feedback"):
                vlm_feedback = node._vlm_feedback
                if isinstance(vlm_feedback, dict):
                    # Look for systemic issues identified by VLM
                    if "systemic_issues" in vlm_feedback:
                        vlm_issues.update(vlm_feedback["systemic_issues"])
                    # Look for recurring patterns in plot analysis
                    if "plot_analyses" in vlm_feedback:
                        for analysis in vlm_feedback["plot_analyses"]:
                            if "limitation" in analysis.get("type", "").lower():
                                vlm_issues.add(
                                    f"VLM (Node {node.id}): {analysis['analysis']}"
                                )

        issues.extend(list(vlm_issues))

        return issues

    def _analyze_progress(self, journal: Journal) -> Dict[str, Any]:
        """Analyze progress and convergence in the current stage"""
        progress = {
            "iterations_completed": len(journal.nodes),
            "improvements_found": 0,
            "convergence_status": "not_converged",
            "improvement_trend": [],
            "recent_changes": [],
        }

        import math

        def objective_value(node: Node) -> Optional[float]:
            if getattr(node, "metric_provenance", None) != "deterministic_verified":
                return None
            metric = getattr(node, "metric", None)
            if metric is None or metric.value is None:
                return None
            try:
                mean_value = float(metric.get_mean_value())
            except Exception:
                return None
            if math.isnan(mean_value) or math.isinf(mean_value):
                return None
            try:
                should_maximize = bool(metric._should_maximize())
            except Exception:
                should_maximize = True
            return mean_value if should_maximize else -mean_value

        # Track a running best objective to detect improvements and convergence.
        running_best = None
        last_improve_index = None
        good_non_seed_nodes = [
            node
            for node in journal.nodes
            if (not node.is_buggy)
            and node.metric_provenance == "deterministic_verified"
            and (not getattr(node, "is_seed_node", False))
            and (not getattr(node, "is_seed_agg_node", False))
        ]
        for index, node in enumerate(good_non_seed_nodes):
            obj = objective_value(node)
            if obj is None:
                continue
            if running_best is None:
                running_best = obj
                last_improve_index = index
                improved = True
            else:
                eps = max(1e-6, 1e-3 * abs(running_best))
                improved = obj > running_best + eps
                if improved:
                    running_best = obj
                    last_improve_index = index
                    progress["improvements_found"] += 1

            progress["improvement_trend"].append(
                {
                    "node_id": node.id,
                    "step": node.step,
                    "objective": obj,
                    "best_objective": running_best,
                    "improved": improved,
                }
            )

        patience = 5
        if (
            last_improve_index is not None
            and len(good_non_seed_nodes) - 1 - last_improve_index >= patience
        ):
            progress["convergence_status"] = "converged"

        # Analyze recent changes
        recent_nodes = (
            good_non_seed_nodes[-3:]
            if len(good_non_seed_nodes) >= 3
            else good_non_seed_nodes
        )
        for node in recent_nodes:
            obj = objective_value(node)
            change = {
                "node_id": node.id,
                "step": node.step,
                "objective": obj,
                "metric": node.metric.value if getattr(node, "metric", None) else None,
                "parent_id": node.parent.id if node.parent else None,
                "analysis": node.analysis if hasattr(node, "analysis") else None,
            }
            progress["recent_changes"].append(change)

        return progress

    def _evaluate_stage_progression(
        self, current_stage: Stage, previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate whether experiment is ready for next stage"""

        eval_prompt = f"""
        Evaluate whether the current experimental stage should progress to the next stage.
        Consider all available evidence holistically:

        Current Stage Information:
        - Name: {current_stage.name}
        - Description: {current_stage.description}
        - Goals: {', '.join(current_stage.goals) if isinstance(current_stage.goals, list) else current_stage.goals}

        Performance Metrics:
        {json.dumps(previous_results.get('metrics', {}), indent=2)}

        Identified Issues:
        {json.dumps(previous_results.get('issues', []), indent=2)}

        Progress Analysis:
        {json.dumps(previous_results.get('progress', {}), indent=2)}

        Expected Stage Progression:
        1. Initial Implementation: Focus on basic working implementation
        2. Baseline Tuning: Systematic optimization of core parameters
        3. Creative Research: Novel improvements and approaches
        4. Ablation Studies: Systematic component analysis

        Consider factors like:
        - Progress toward stage goals
        - Performance trends and stability
        - Quality and reliability of results
        - Understanding of the problem
        - Presence of systematic issues
        - Convergence indicators
        - Readiness for next stage challenges

        Provide a holistic evaluation of whether the experiment should:
        1. Progress to next stage
        2. Continue current stage with specific focus
        3. Extend current stage with modifications
        """

        try:
            evaluation = query(
                system_message=eval_prompt,
                user_message=None,
                func_spec=stage_progress_eval_spec,
                model=self.cfg.agent.feedback.model,
                temperature=self.cfg.agent.feedback.temp,
                max_tokens=_stage_max_tokens(self.cfg.agent.feedback),
            )

            # Log the evaluation for transparency
            logger.info(
                "Stage progression evaluation received: ready=%s recommendations=%d",
                evaluation.get("ready_for_next_stage"),
                len(evaluation.get("recommendations") or []),
            )

            return evaluation

        except Exception as e:
            if isinstance(e, ResearchDecisionError) or is_llm_budget_exception(e):
                raise
            _fail_research_decision("stage progression evaluation", e)
