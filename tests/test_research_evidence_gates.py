from __future__ import annotations

import copy
import json
import math
import statistics
from concurrent.futures import Future
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from ai_scientist.resources import resolve_bfts_config_path
from ai_scientist.treesearch.agent_manager import (
    AgentManager,
    ExperimentCannotContinueError,
    Stage,
    _sha256_json,
)
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import (
    HyperparamTuningIdea,
    MinimalAgent,
    ParallelAgent,
    _ablation_component_was_transformed,
    _ablation_idea_key,
    _canonical_idea_key,
    _configured_multi_seed_values,
    _inject_seed_bootstrap,
    _semantic_code_hash,
    _validate_confirmation_seed_set,
    get_gpu_devices,
    metric_selection_spec,
)
from ai_scientist.treesearch.interpreter import ExecutionResult
from ai_scientist.treesearch.utils.metric import MetricValue
from ai_scientist.utils.deterministic_evaluator import evaluate_experiment_data

TASK = json.dumps(
    {
        "Title": "Evidence gate",
        "Abstract": "Exercise deterministic research progression.",
        "Short Hypothesis": "A bounded candidate improves a locked control.",
        "Experiments": [],
        "Risk Factors and Limitations": [],
    }
)

CONTROL_CODE = """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
data_rng = random.Random(XSCIENTIST_DATA_SEED)
training_rng = random.Random(XSCIENTIST_TRAINING_SEED)
print(data_rng.random(), training_rng.random())
"""


def _manager(tmp_path: Path, *, workers: int = 1) -> AgentManager:
    workspace = tmp_path / "workspaces" / "0-run"
    log_dir = tmp_path / "logs" / "0-run"
    workspace.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    cfg = OmegaConf.load(resolve_bfts_config_path("bfts_config.yaml"))
    cfg.workspace_dir = workspace
    cfg.log_dir = log_dir
    cfg.agent.num_workers = workers
    return AgentManager(task_desc=TASK, cfg=cfg, workspace_dir=workspace)


def _predictions(correct: int) -> list[int]:
    labels = [0, 1, 0, 1]
    result = list(labels)
    for index in range(4 - correct):
        result[-(index + 1)] = 1 - result[-(index + 1)]
    return result


def _payload(correct_by_dataset: tuple[int, int, int]) -> dict:
    return {
        f"dataset_{index}": {
            "evaluation_inputs": [
                [index, sample, float(index + sample)] for sample in range(4)
            ],
            "sample_ids": [f"{index}-{sample}" for sample in range(4)],
            "ground_truth": [0, 1, 0, 1],
            "predictions": _predictions(correct),
        }
        for index, correct in enumerate(correct_by_dataset)
    }


def _attach_evidence(
    manager: AgentManager,
    node: Node,
    *,
    correct_by_dataset: tuple[int, int, int],
    metric: str = "accuracy",
) -> Node:
    evidence_dir = (
        Path(manager.cfg.log_dir) / "experiment_results" / f"experiment_{node.id}"
    )
    evidence_dir.mkdir(parents=True)
    artifact = evidence_dir / "experiment_data.npy"
    np.save(artifact, _payload(correct_by_dataset))
    (evidence_dir / "experiment_code.py").write_text(node.code, encoding="utf-8")
    report = evaluate_experiment_data(artifact, requested_metric=metric)
    assert report["status"] == "verified"
    node.metric = MetricValue(report["metric"])
    node.metric_provenance = "deterministic_verified"
    node.evaluation_report = report
    node.exp_results_dir = str(evidence_dir.resolve())
    node.is_buggy = False
    node.is_buggy_plots = False
    return node


def _finalize_stage1(
    manager: AgentManager,
    *,
    correct_by_dataset: tuple[int, int, int] = (2, 2, 2),
) -> Node:
    stage = manager.current_stage
    stage.attempt_count = max(stage.attempt_count, 1)
    stage.evaluation_metric = "accuracy"
    control = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE, plan="control"),
        correct_by_dataset=correct_by_dataset,
    )
    manager.journals[stage.name].append(control)
    complete, reason = manager._check_stage_completion(stage)
    assert complete is False
    assert "multi-seed evidence pending" in reason

    seed_nodes = []
    for seed in manager._configured_multi_seed_seeds():
        seeded_code, bootstrap_hash = _inject_seed_bootstrap(control.code, seed)
        seed_node = _attach_evidence(
            manager,
            Node(
                code=seeded_code,
                plan="seed",
                parent=control,
                is_seed_node=True,
                random_seed=seed,
                seed_bootstrap_hash=bootstrap_hash,
            ),
            correct_by_dataset=correct_by_dataset,
        )
        manager.journals[stage.name].append(seed_node)
        seed_nodes.append(seed_node)
    manager._finalize_multi_seed_gate(
        stage,
        manager._get_best_implementation(stage.name, require_multi_seed=False),
        seed_nodes,
    )
    assert manager._check_stage_completion(stage)[0] is True
    return control


def test_seed_rewrite_changes_only_explicit_training_role() -> None:
    source = (
        "from __future__ import annotations\n"
        + CONTROL_CODE
        + "class Custom:\n"
        + "    def seed(self, value):\n"
        + "        return value\n"
        + "print(Custom().seed(99))\n"
    )
    seeded, receipt = _inject_seed_bootstrap(source, 123)

    assert "XSCIENTIST_DATA_SEED = 7" in seeded
    assert "XSCIENTIST_TRAINING_SEED = 123" in seeded
    assert ".seed(99)" in seeded
    assert receipt.startswith("sha256:")
    with pytest.raises(ExperimentCannotContinueError, match="seed roles"):
        _inject_seed_bootstrap("print('no roles')", 1)


def test_confirmation_seeds_are_held_out_from_selection_seed() -> None:
    _validate_confirmation_seed_set(CONTROL_CODE, [123, 456, 789])

    with pytest.raises(ExperimentCannotContinueError, match="held out"):
        _validate_confirmation_seed_set(CONTROL_CODE, [42, 123, 456])


@pytest.mark.parametrize(
    "source",
    [
        """\
XSCIENTIST_DATA_SEED = XSCIENTIST_TRAINING_SEED = 42
import random
random.Random(XSCIENTIST_DATA_SEED)
random.Random(XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = SPLIT_SEED = 42
import random
random.Random(XSCIENTIST_DATA_SEED)
random.Random(XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
XSCIENTIST_TRAINING_SEED += 1
import random
random.Random(XSCIENTIST_DATA_SEED)
random.Random(XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
random.Random(XSCIENTIST_DATA_SEED)
print(XSCIENTIST_TRAINING_SEED)
random.Random(999)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
def ignore(*, seed):
    return seed
ignore(seed=XSCIENTIST_DATA_SEED)
ignore(seed=XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
random.Random(XSCIENTIST_DATA_SEED)
random.Random(XSCIENTIST_TRAINING_SEED * 0)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
random.Random(XSCIENTIST_DATA_SEED)
random.seed(XSCIENTIST_TRAINING_SEED)
random.seed(0)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
if 0:
    random.Random(XSCIENTIST_DATA_SEED)
    random.Random(XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random as rng
rng = object()
random.Random(XSCIENTIST_DATA_SEED)
random.Random(XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
FLAG = True
import random
if FLAG:
    random.Random(XSCIENTIST_DATA_SEED)
else:
    random.Random(XSCIENTIST_TRAINING_SEED)
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
def configure():
    return
    random.Random(XSCIENTIST_DATA_SEED)
    random.Random(XSCIENTIST_TRAINING_SEED)
configure()
""",
        """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
FLAG = True
import random
def configure():
    if FLAG:
        return
    random.Random(XSCIENTIST_DATA_SEED)
    random.Random(XSCIENTIST_TRAINING_SEED)
configure()
""",
    ],
)
def test_seed_rewrite_rejects_ambiguous_or_noop_training_bindings(
    source: str,
) -> None:
    with pytest.raises(ExperimentCannotContinueError):
        _inject_seed_bootstrap(source, 123)


def test_idea_identity_is_unicode_safe_and_ablation_component_bound() -> None:
    assert _canonical_idea_key("  学习率：调整  ") == "学习率 调整"
    assert _canonical_idea_key("Drop-out / RATE") == "drop out rate"
    assert _ablation_idea_key("移除 Dropout", "Dropout 层") == _ablation_idea_key(
        "禁用 Dropout",
        "dropout 层",
    )


def test_glm_execution_review_is_advisory_not_a_scientific_veto(
    tmp_path: Path,
) -> None:
    cfg = _manager(tmp_path).cfg
    agent = MinimalAgent("task", cfg)
    node = Node(code="print('ok')")
    successful = ExecutionResult(term_out=["ok\n"], exec_time=0.1, exc_type=None)

    with mock.patch(
        "ai_scientist.treesearch.parallel_agent.query",
        return_value={"is_bug": True, "summary": "agent dislikes the result"},
    ):
        agent.parse_exec_result(node, successful, str(tmp_path))

    assert node.agent_review_bug_advisory is True
    assert node.is_buggy is False

    failed = ExecutionResult(
        term_out=[],
        exec_time=0.1,
        exc_type="RuntimeError",
    )
    with mock.patch(
        "ai_scientist.treesearch.parallel_agent.query",
        return_value={"is_bug": False, "summary": "agent says it is fine"},
    ):
        agent.parse_exec_result(node, failed, str(tmp_path))
    assert node.agent_review_bug_advisory is False
    assert node.is_buggy is True


def test_confirmation_seed_worker_uses_host_only_verification(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    seeded_code, bootstrap_hash = _inject_seed_bootstrap(CONTROL_CODE, 123)
    parent = Node(
        code=seeded_code,
        random_seed=123,
        seed_bootstrap_hash=bootstrap_hash,
        is_buggy=False,
    )

    class FakeInterpreter:
        def __init__(self, workspace: str | Path) -> None:
            self.workspace = Path(workspace)

        def run(self, _code: str, _reset: bool) -> ExecutionResult:
            working = self.workspace / "working"
            working.mkdir(parents=True, exist_ok=True)
            np.save(working / "experiment_data.npy", _payload((2, 2, 2)))
            return ExecutionResult(
                term_out=["seed complete\n"],
                exec_time=0.1,
                exc_type=None,
            )

        def cleanup_session(self) -> None:
            return None

    with (
        mock.patch(
            "ai_scientist.treesearch.parallel_agent._interpreter_for_workspace",
            side_effect=lambda _cfg, workspace, **_kwargs: FakeInterpreter(workspace),
        ),
        mock.patch(
            "ai_scientist.treesearch.parallel_agent.query",
            side_effect=AssertionError("seed replay must not call an LLM"),
        ) as llm_query,
    ):
        result = ParallelAgent._process_node_wrapper(
            parent.to_dict(),
            "task",
            manager.cfg,
            None,
            "",
            "accuracy",
            manager.current_stage.name,
            seed_eval=True,
        )

    assert llm_query.call_count == 0
    assert result["metric_provenance"] == "deterministic_verified"
    assert result["is_buggy"] is False
    assert result["agent_review_bug_advisory"] is None
    assert result["plots"] == []


def test_metric_selection_and_worker_count_are_host_bounded(tmp_path: Path) -> None:
    assert set(metric_selection_spec.json_schema["properties"]["metric"]["enum"]) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "rmse",
        "mse",
        "mae",
        "r2",
    }
    cfg = _manager(tmp_path).cfg
    for invalid in (True, 0, 65, 1.5, "2"):
        cfg.agent.num_workers = invalid
        with pytest.raises(ValueError, match="num_workers"):
            ParallelAgent(
                task_desc="task",
                cfg=cfg,
                journal=Journal(),
                evaluation_metric="accuracy",
            )


def test_multi_seed_config_is_bounded_before_materialization() -> None:
    from types import SimpleNamespace

    with pytest.raises(ExperimentCannotContinueError, match="between 3 and 32"):
        _configured_multi_seed_values(SimpleNamespace(seeds=None, num_seeds=10**12))
    with pytest.raises(ExperimentCannotContinueError, match="bounded integers"):
        _configured_multi_seed_values(
            SimpleNamespace(seeds=[[1], [2], [3]], num_seeds=3)
        )
    with pytest.raises(ExperimentCannotContinueError, match="sequence"):
        _configured_multi_seed_values(
            SimpleNamespace(seeds=(value for value in [1, 2, 3]), num_seeds=3)
        )


@pytest.mark.parametrize(
    ("visible", "expected"),
    [
        ("2,3", ["2", "3"]),
        ("-1", []),
        ("", []),
        ("GPU-deadbeef,MIG-cafebabe", ["GPU-deadbeef", "MIG-cafebabe"]),
    ],
)
def test_gpu_discovery_preserves_authorized_cuda_tokens(
    monkeypatch: pytest.MonkeyPatch,
    visible: str,
    expected: list[str],
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    assert get_gpu_devices() == expected


def test_ablation_semantic_identity_rejects_comments_and_dead_code() -> None:
    control = "DROPOUT_RATE = 0.5\nconfig = {'dropout': DROPOUT_RATE}\n"
    comment_only = control + "# dropout removed\n"
    dead_code = control + "if False:\n    DROPOUT_RATE = 0.0\n"
    real_disable = "DROPOUT_RATE = 0.0\nconfig = {'dropout': DROPOUT_RATE}\n"

    assert _semantic_code_hash(comment_only) == _semantic_code_hash(control)
    assert _semantic_code_hash(dead_code) == _semantic_code_hash(control)
    assert not _ablation_component_was_transformed(
        control,
        comment_only,
        "dropout component",
    )
    assert not _ablation_component_was_transformed(
        control,
        dead_code,
        "dropout component",
    )
    assert _semantic_code_hash(real_disable) != _semantic_code_hash(control)
    assert _ablation_component_was_transformed(
        control,
        real_disable,
        "dropout component",
    )


def test_finalized_stage_replays_and_transitions_from_checkpoint(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    stage1 = manager.current_stage
    _finalize_stage1(manager)
    stage1.attempt_count = stage1.max_iterations

    checkpoint = manager._save_checkpoint()
    restored = AgentManager.from_checkpoint(
        checkpoint,
        cfg=manager.cfg,
        workspace_dir=manager.workspace_dir,
    )
    assert restored._check_stage_completion(restored.current_stage)[0] is True

    next_stage = restored._advance_main_stage()
    assert next_stage is not None
    assert next_stage.stage_number == 2
    assert next_stage.evaluation_metric == "accuracy"
    assert restored.completed_stages == [stage1.name]


def test_checkpoint_replays_confined_cwd_relative_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    manager = _manager(Path("."))
    _finalize_stage1(manager)

    checkpoint = manager._save_checkpoint()
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    nodes = envelope["payload"]["journals"][manager.current_stage.name]["nodes"]
    evidence_paths = [node["exp_results_dir"] for node in nodes]
    assert evidence_paths
    assert all(not Path(path).is_absolute() for path in evidence_paths)

    restored = AgentManager.from_checkpoint(
        checkpoint,
        cfg=manager.cfg,
        workspace_dir=manager.workspace_dir,
    )
    assert restored._check_stage_completion(restored.current_stage)[0] is True


def test_checkpoint_artifact_locator_is_independent_of_resume_cwd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(tmp_path / "run")
    _finalize_stage1(manager)
    checkpoint = manager._save_checkpoint()
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    nodes = envelope["payload"]["journals"][manager.current_stage.name]["nodes"]
    assert all(not Path(node["exp_results_dir"]).is_absolute() for node in nodes)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    restored = AgentManager.from_checkpoint(
        checkpoint,
        cfg=manager.cfg,
        workspace_dir=manager.workspace_dir,
    )

    assert restored._check_stage_completion(restored.current_stage)[0] is True


def test_checkpoint_rejects_path_unsafe_node_identity(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    node = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    manager.journals[stage.name].append(node)
    stage.attempt_count = 1
    checkpoint = manager._save_checkpoint()
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    serialized = envelope["payload"]["journals"][stage.name]["nodes"][0]
    serialized["id"] = "anchor/../../outside"
    envelope["payload_hash"] = _sha256_json(envelope["payload"])
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid id"):
        AgentManager.from_checkpoint(
            checkpoint,
            cfg=manager.cfg,
            workspace_dir=manager.workspace_dir,
        )


def test_checkpoint_replay_rejects_symlinked_evidence_directory(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    node = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    manager.journals[stage.name].append(node)
    stage.attempt_count = 1
    checkpoint = manager._save_checkpoint()
    evidence_dir = Path(node.exp_results_dir)
    relocated = tmp_path / "relocated-evidence"
    evidence_dir.rename(relocated)
    try:
        evidence_dir.symlink_to(relocated, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")

    with pytest.raises(ValueError, match="scientific evidence is inconsistent"):
        AgentManager.from_checkpoint(
            checkpoint,
            cfg=manager.cfg,
            workspace_dir=manager.workspace_dir,
        )


def test_checkpoint_replay_rejects_artifact_tampering(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    control = _finalize_stage1(manager)
    checkpoint = manager._save_checkpoint()
    artifact = Path(control.exp_results_dir) / "experiment_data.npy"
    np.save(artifact, _payload((4, 4, 4)))

    with pytest.raises(ValueError, match="scientific evidence is inconsistent"):
        AgentManager.from_checkpoint(
            checkpoint,
            cfg=manager.cfg,
            workspace_dir=manager.workspace_dir,
        )


def test_checkpoint_replay_rejects_forged_evaluator_identity(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    node = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    manager.journals[stage.name].append(node)
    checkpoint = manager._save_checkpoint()
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    serialized = envelope["payload"]["journals"][stage.name]["nodes"][0]
    report = serialized["evaluation_report"]
    report["evaluator_hash"] = "sha256:" + "f" * 64
    report_without_hash = dict(report)
    report_without_hash.pop("result_hash")
    report["result_hash"] = _sha256_json(report_without_hash)
    envelope["payload_hash"] = _sha256_json(envelope["payload"])
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific evidence is inconsistent"):
        AgentManager.from_checkpoint(
            checkpoint,
            cfg=manager.cfg,
            workspace_dir=manager.workspace_dir,
        )


def test_checkpoint_cannot_claim_terminal_after_only_stage1(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    _finalize_stage1(manager)
    checkpoint = manager._save_checkpoint()
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    stage_name = manager.current_stage.name
    envelope["payload"]["current_stage"] = None
    envelope["payload"]["completed_stages"] = [stage_name]
    envelope["payload_hash"] = _sha256_json(envelope["payload"])
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="complete every main stage"):
        AgentManager.from_checkpoint(
            checkpoint,
            cfg=manager.cfg,
            workspace_dir=manager.workspace_dir,
        )


def test_checkpoint_cannot_replenish_spent_attempt_budget(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    node = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    manager.journals[stage.name].append(node)
    stage.attempt_count = 1
    checkpoint = manager._save_checkpoint()
    envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
    envelope["payload"]["stages"][0]["attempt_count"] = 0
    envelope["payload"]["current_stage"]["attempt_count"] = 0
    envelope["payload_hash"] = _sha256_json(envelope["payload"])
    checkpoint.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ValueError, match="scientific evidence is inconsistent"):
        AgentManager.from_checkpoint(
            checkpoint,
            cfg=manager.cfg,
            workspace_dir=manager.workspace_dir,
        )


def test_checkpoint_allows_reserved_attempt_without_result(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.current_stage.attempt_count = 1
    checkpoint = manager._save_checkpoint()

    restored = AgentManager.from_checkpoint(
        checkpoint,
        cfg=manager.cfg,
        workspace_dir=manager.workspace_dir,
    )

    assert restored.current_stage.attempt_count == 1
    assert restored.journals[restored.current_stage.name].nodes == []


def test_checkpoint_restores_locked_candidate_pending_seed_receipt(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    candidate = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    manager.journals[stage.name].append(candidate)
    stage.attempt_count = 1
    complete, reason = manager._check_stage_completion(stage)
    assert complete is False
    assert "multi-seed evidence pending" in reason
    assert stage.qualified_node_id == candidate.id

    restored = AgentManager.from_checkpoint(
        manager._save_checkpoint(),
        cfg=manager.cfg,
        workspace_dir=manager.workspace_dir,
    )

    assert restored.current_stage.qualified_node_id == candidate.id
    assert restored.current_stage.multi_seed_receipt_hash is None
    assert restored._check_stage_completion(restored.current_stage)[0] is False


def test_receipt_rejects_missing_seed_journal_member(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    control = _finalize_stage1(manager)
    report = control.multi_seed_report
    missing_id = report["seeds"][-1]["node_id"]
    journal = manager.journals[manager.current_stage.name]
    missing = journal.get_node_by_id(missing_id)
    journal.nodes.remove(missing)
    control.children.remove(missing)

    with pytest.raises(
        ExperimentCannotContinueError,
        match="does not resolve to journal evidence|not claimed",
    ):
        manager._check_stage_completion(manager.current_stage)


def test_receipt_rejects_forged_seed_bootstrap_member(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    control = _finalize_stage1(manager)
    report = control.multi_seed_report
    report["seeds"][0]["seed_bootstrap_hash"] = "sha256:" + "0" * 64
    unsigned = copy.deepcopy(report)
    unsigned.pop("receipt_hash")
    report["receipt_hash"] = _sha256_json(unsigned)
    manager.current_stage.multi_seed_receipt_hash = report["receipt_hash"]

    with pytest.raises(
        ExperimentCannotContinueError,
        match="does not resolve to journal evidence",
    ):
        manager._check_stage_completion(manager.current_stage)


def test_stage2_ignores_off_metric_candidate_and_requires_paired_improvement(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    _finalize_stage1(manager, correct_by_dataset=(2, 2, 2))
    stage2 = manager._advance_main_stage()
    baseline = manager._get_best_implementation(manager.completed_stages[-1])
    manager.journals[stage2.name].append(baseline)

    off_metric = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE + "\n# off metric", parent=baseline),
        correct_by_dataset=(4, 4, 4),
        metric="f1",
    )
    candidate_code = CONTROL_CODE + "\nLEARNING_RATE = 0.01\n"
    candidate = _attach_evidence(
        manager,
        Node(
            code=candidate_code,
            parent=baseline,
            hyperparam_name="learning_rate",
        ),
        correct_by_dataset=(3, 3, 2),
    )
    manager.journals[stage2.name].append(off_metric)
    manager.journals[stage2.name].append(candidate)

    complete, reason = manager._check_stage_completion(stage2)
    assert complete is False
    assert "multi-seed evidence pending" in reason
    assert stage2.qualified_node_id == candidate.id

    seed_nodes = []
    for seed in manager._configured_multi_seed_seeds():
        seeded_code, bootstrap_hash = _inject_seed_bootstrap(candidate.code, seed)
        seed_node = _attach_evidence(
            manager,
            Node(
                code=seeded_code,
                parent=candidate,
                is_seed_node=True,
                random_seed=seed,
                seed_bootstrap_hash=bootstrap_hash,
            ),
            correct_by_dataset=(3, 3, 2),
        )
        manager.journals[stage2.name].append(seed_node)
        seed_nodes.append(seed_node)
    manager._finalize_multi_seed_gate(
        stage2,
        manager._get_best_implementation(stage2.name, require_multi_seed=False),
        seed_nodes,
    )
    assert manager._check_stage_completion(stage2)[0] is True


def test_run_reserves_only_remaining_attempts(tmp_path: Path) -> None:
    manager = _manager(tmp_path, workers=4)
    stage = manager.current_stage
    stage.max_iterations = 1

    class FakeAgent:
        num_workers = 4

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def step(self, _callback, *, max_new_nodes: int) -> int:
            self.batch_sizes.append(max_new_nodes)
            manager.journals[stage.name].append(
                Node(code="raise RuntimeError()", is_buggy=True)
            )
            return max_new_nodes

    fake = FakeAgent()
    with (
        mock.patch.object(manager, "_create_agent_for_stage", return_value=fake),
        mock.patch.object(manager, "_save_checkpoint", return_value=None),
        pytest.raises(ExperimentCannotContinueError, match="exhausted"),
    ):
        manager.run(lambda _code, _reset: None)
    assert fake.batch_sizes == [1]
    assert stage.attempt_count == 1


def test_candidate_lock_is_checkpointed_before_seed_submission(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    candidate = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    manager.journals[stage.name].append(candidate)
    stage.attempt_count = 1
    observed: dict[str, object] = {}

    class FakeAgent:
        num_workers = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def _run_multi_seed_evaluation(self, _qualified):
            checkpoint = (
                manager._artifact_root / f"stage_{stage.name}" / "checkpoint.json"
            )
            envelope = json.loads(checkpoint.read_text(encoding="utf-8"))
            persisted = envelope["payload"]["current_stage"]
            observed["qualified_node_id"] = persisted["qualified_node_id"]
            observed["evaluation_metric"] = persisted["evaluation_metric"]
            observed["multi_seed_receipt_hash"] = persisted["multi_seed_receipt_hash"]
            raise RuntimeError("seed worker stopped")

    with (
        mock.patch.object(manager, "_create_agent_for_stage", return_value=FakeAgent()),
        pytest.raises(RuntimeError, match="seed worker stopped"),
    ):
        manager.run(lambda _code, _reset: None)

    assert observed == {
        "qualified_node_id": candidate.id,
        "evaluation_metric": "accuracy",
        "multi_seed_receipt_hash": None,
    }


def test_rejected_seed_batch_is_preserved_with_student_t_uncertainty(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    stage = manager.current_stage
    candidate = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(3, 3, 3),
    )
    manager.journals[stage.name].append(candidate)
    stage.attempt_count = stage.max_iterations

    class FakeAgent:
        num_workers = 1

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def _run_multi_seed_evaluation(self, qualified):
            locked = manager.journals[stage.name].get_node_by_id(qualified.id)
            nodes = []
            for seed, correct in zip(
                manager._configured_multi_seed_seeds(),
                (2, 3, 3),
            ):
                seeded_code, bootstrap_hash = _inject_seed_bootstrap(
                    locked.code,
                    seed,
                )
                node = _attach_evidence(
                    manager,
                    Node(
                        code=seeded_code,
                        parent=locked,
                        is_seed_node=True,
                        random_seed=seed,
                        seed_bootstrap_hash=bootstrap_hash,
                    ),
                    correct_by_dataset=(correct, correct, correct),
                )
                manager.journals[stage.name].append(node)
                nodes.append(node)
            return nodes

    with (
        mock.patch.object(manager, "_create_agent_for_stage", return_value=FakeAgent()),
        pytest.raises(ExperimentCannotContinueError, match="exhausted"),
    ):
        manager.run(lambda _code, _reset: None)

    assert stage.qualified_node_id is None
    assert stage.multi_seed_receipt_hash is None
    assert candidate.multi_seed_report is None
    assert len(candidate.multi_seed_attempts) == 1
    attempt = candidate.multi_seed_attempts[0]
    assert attempt["reason_code"] == "stability"
    values = [0.5, 0.75, 0.75]
    expected_se = statistics.stdev(values) / math.sqrt(3)
    assert attempt["datasets"]["dataset_0"]["ci95_half_width"] == pytest.approx(
        4.303 * expected_se
    )
    assert (
        len([node for node in manager.journals[stage.name].nodes if node.is_seed_node])
        == 3
    )

    checkpoint = manager._save_checkpoint()
    restored = AgentManager.from_checkpoint(
        checkpoint,
        cfg=manager.cfg,
        workspace_dir=manager.workspace_dir,
    )
    restored_candidate = restored.journals[stage.name].get_node_by_id(candidate.id)
    assert restored_candidate.multi_seed_attempts == candidate.multi_seed_attempts


def test_parallel_step_does_not_commit_partial_failed_batch(tmp_path: Path) -> None:
    manager = _manager(tmp_path, workers=3)
    agent = ParallelAgent.__new__(ParallelAgent)
    agent.num_workers = 3
    agent.journal = Journal()
    agent.cfg = manager.cfg
    agent.timeout = 1
    agent.gpu_manager = None
    agent.task_desc = "task"
    agent.evaluation_metrics = "accuracy"
    agent.stage_name = manager.current_stage.name
    agent.best_stage1_node = None
    agent.best_stage2_node = None
    agent.best_stage3_node = None
    agent._is_shutdown = False
    agent._hyperparam_tuning_state = {"tried_hyperparams": set()}
    agent._ablation_state = {"completed_ablations": set()}

    results: list[Future] = []
    for index in range(2):
        future = Future()
        future.set_result(Node(code=f"print({index})", is_buggy=True).to_dict())
        results.append(future)
    failed = Future()
    failed.set_exception(RuntimeError("worker details must not be persisted"))
    results.append(failed)

    class FakeExecutor:
        def submit(self, *_args, **_kwargs):
            return results.pop(0)

        def shutdown(self, **_kwargs):
            return None

    agent.executor = FakeExecutor()
    with (
        mock.patch.object(
            agent,
            "_select_parallel_nodes",
            return_value=[None, None, None],
        ),
        pytest.raises(ExperimentCannotContinueError, match="atomically"),
    ):
        agent.step(lambda _code, _reset: None, max_new_nodes=3)
    assert agent.journal.nodes == []


def test_multi_seed_evaluation_runs_bounded_gpu_waves(tmp_path: Path) -> None:
    manager = _manager(tmp_path, workers=2)
    manager.cfg.agent.multi_seed_eval.seeds = [1, 2, 3, 4, 5]
    parent = _attach_evidence(
        manager,
        Node(code=CONTROL_CODE),
        correct_by_dataset=(2, 2, 2),
    )
    journal = Journal()
    journal.append(parent)

    events: list[tuple[str, str]] = []

    class FakeGPUManager:
        def __init__(self) -> None:
            self.available = ["gpu0", "gpu1"]
            self.assignments: dict[str, str] = {}

        def acquire_gpu(self, process_id: str) -> str:
            if not self.available:
                raise RuntimeError("no GPU")
            gpu = self.available.pop(0)
            self.assignments[process_id] = gpu
            events.append(("acquire", process_id))
            return gpu

        def release_gpu(self, process_id: str) -> None:
            gpu = self.assignments.pop(process_id)
            self.available.append(gpu)
            events.append(("release", process_id))

    class FakeExecutor:
        def submit(self, _callable, seed_node_data, *_args):
            payload = copy.deepcopy(seed_node_data)
            seed = payload["random_seed"]
            payload["id"] = f"seed_{seed}"
            payload["parent_id"] = parent.id
            payload["is_seed_node"] = True
            payload["is_seed_agg_node"] = False
            future = Future()
            future.set_result(payload)
            return future

    agent = ParallelAgent.__new__(ParallelAgent)
    agent.num_workers = 2
    agent.journal = journal
    agent.cfg = manager.cfg
    agent.timeout = 1
    agent.gpu_manager = FakeGPUManager()
    agent.task_desc = "task"
    agent.evaluation_metrics = "accuracy"
    agent.stage_name = manager.current_stage.name
    agent.executor = FakeExecutor()
    agent._is_shutdown = False

    seeds = agent._run_multi_seed_evaluation(parent)

    assert [node.random_seed for node in seeds] == [1, 2, 3, 4, 5]
    assert {node.id for node in parent.children} == {
        "seed_1",
        "seed_2",
        "seed_3",
        "seed_4",
        "seed_5",
    }
    assert events == [
        ("acquire", "seed_1_worker"),
        ("acquire", "seed_2_worker"),
        ("release", "seed_1_worker"),
        ("release", "seed_2_worker"),
        ("acquire", "seed_3_worker"),
        ("acquire", "seed_4_worker"),
        ("release", "seed_3_worker"),
        ("release", "seed_4_worker"),
        ("acquire", "seed_5_worker"),
        ("release", "seed_5_worker"),
    ]


def test_final_report_view_excludes_exploratory_and_rejected_nodes() -> None:
    manager = AgentManager.__new__(AgentManager)
    manager.main_stage_dict = {1: "one", 2: "two", 3: "three", 4: "four"}
    manager.stages = []
    manager.completed_stages = []
    manager.journals = {}

    for stage_number in range(1, 5):
        stage_name = f"{stage_number}_stage_1_final"
        receipt_hash = f"receipt-{stage_number}"
        qualified = Node(code="qualified", id=f"qualified_{stage_number}")
        seed = Node(
            code="seed",
            id=f"seed_{stage_number}",
            parent=qualified,
            is_seed_node=True,
        )
        exploratory = Node(code="exploratory", id=f"explore_{stage_number}")
        qualified.multi_seed_report = {
            "stage": stage_name,
            "receipt_hash": receipt_hash,
            "seeds": [{"node_id": seed.id}],
        }
        journal = Journal()
        journal.append(qualified)
        journal.append(seed)
        journal.append(exploratory)
        stage = Stage(
            name=stage_name,
            description="final",
            goals="gate",
            max_iterations=1,
            num_drafts=0,
            stage_number=stage_number,
            qualified_node_id=qualified.id,
            multi_seed_receipt_hash=receipt_hash,
        )
        manager.stages.append(stage)
        manager.completed_stages.append(stage_name)
        manager.journals[stage_name] = journal

    manager._validate_report_origin = mock.Mock(
        side_effect=lambda node: node.multi_seed_report
    )

    report_journals = manager.qualified_report_journals()

    assert len(report_journals) == 4
    for stage_number, (_stage_name, journal) in enumerate(report_journals, start=1):
        assert [node.id for node in journal.nodes] == [
            f"qualified_{stage_number}",
            f"seed_{stage_number}",
        ]
        assert journal.nodes[1].parent is journal.nodes[0]


def test_parallel_step_reserves_distinct_batch_ideas(tmp_path: Path) -> None:
    manager = _manager(tmp_path, workers=2)
    parent = Node(code="print('base')", is_buggy=False, id="baseline")
    journal = Journal()
    journal.append(parent)
    agent = ParallelAgent.__new__(ParallelAgent)
    agent.num_workers = 2
    agent.journal = journal
    agent.cfg = manager.cfg
    agent.timeout = 1
    agent.gpu_manager = None
    agent.task_desc = "task"
    agent.evaluation_metrics = "accuracy"
    agent.stage_name = "2_baseline_tuning_1_first_attempt"
    agent.best_stage1_node = parent
    agent.best_stage2_node = None
    agent.best_stage3_node = None
    agent._is_shutdown = False
    agent._hyperparam_tuning_state = {"tried_hyperparams": set()}
    agent._ablation_state = {"completed_ablations": set()}

    class FakeExecutor:
        def __init__(self) -> None:
            self.index = 0

        def submit(self, *args):
            node_data = copy.deepcopy(args[1])
            idea = args[9]
            node_data["id"] = f"candidate_{self.index}"
            node_data["parent_id"] = parent.id
            node_data["hyperparam_name"] = idea.name
            node_data["is_buggy"] = False
            self.index += 1
            future = Future()
            future.set_result(node_data)
            return future

    agent.executor = FakeExecutor()
    proposals = [
        HyperparamTuningIdea("Learning Rate", "first"),
        HyperparamTuningIdea("learning-rate", "duplicate"),
        HyperparamTuningIdea("Batch Size", "second"),
    ]
    with (
        mock.patch.object(
            agent,
            "_select_parallel_nodes",
            return_value=[parent, parent],
        ),
        mock.patch.object(
            agent,
            "_generate_hyperparam_tuning_idea",
            side_effect=proposals,
        ) as generate,
        mock.patch.object(Journal, "generate_summary", return_value="summary"),
    ):
        committed = agent.step(lambda _code, _reset: None, max_new_nodes=2)

    assert committed == 2
    assert generate.call_count == 3
    assert {node.hyperparam_name for node in journal.nodes[1:]} == {
        "Learning Rate",
        "Batch Size",
    }
    assert agent._hyperparam_tuning_state["tried_hyperparams"] == {
        "learning rate",
        "batch size",
    }


def test_parallel_agent_rebuilds_all_attempted_idea_keys_on_resume(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path, workers=1)
    failed_hyperparam = Node(
        code="failed",
        is_buggy=True,
        hyperparam_name="学习率：调整",
    )
    failed_ablation = Node(
        code="failed",
        is_buggy=True,
        ablation_name="移除正则",
        ablation_component="Dropout 层",
    )
    journal = Journal(nodes=[failed_hyperparam, failed_ablation])

    class FakeExecutor:
        def __init__(self, **_kwargs):
            pass

    with (
        mock.patch(
            "ai_scientist.treesearch.parallel_agent.get_gpu_devices",
            return_value=[],
        ),
        mock.patch(
            "ai_scientist.treesearch.parallel_agent.ProcessPoolExecutor",
            FakeExecutor,
        ),
    ):
        agent = ParallelAgent(
            task_desc="task",
            cfg=manager.cfg,
            journal=journal,
            evaluation_metric="accuracy",
        )

    assert agent._hyperparam_tuning_state["tried_hyperparams"] == {"学习率 调整"}
    assert agent._ablation_state["completed_ablations"] == {"dropout 层"}
