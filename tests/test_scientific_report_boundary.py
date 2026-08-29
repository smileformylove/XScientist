from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from ai_scientist.treesearch import log_summarization
from ai_scientist.treesearch.agent_manager import (
    _build_multi_seed_report,
    _strict_sha256_json,
)
from ai_scientist.treesearch.journal import Journal, Node
from ai_scientist.treesearch.parallel_agent import _inject_seed_bootstrap
from ai_scientist.treesearch.utils.metric import MetricValue
from ai_scientist.utils.deterministic_evaluator import evaluate_experiment_data

CONTROL_CODE = """\
XSCIENTIST_DATA_SEED = 7
XSCIENTIST_TRAINING_SEED = 42
import random
data_rng = random.Random(XSCIENTIST_DATA_SEED)
training_rng = random.Random(XSCIENTIST_TRAINING_SEED)
"""
CONFIRMATION_SEEDS = [123, 456, 789]


def _payload() -> dict:
    labels = [0, 1, 0, 1]
    return {
        "dataset_alpha": {
            "evaluation_inputs": [[index, float(index)] for index in range(4)],
            "sample_ids": [f"alpha-{index}" for index in range(4)],
            "ground_truth": labels,
            "predictions": [0, 1, 0, 0],
        },
        "dataset_beta": {
            "evaluation_inputs": [[index, float(index + 1)] for index in range(4)],
            "sample_ids": [f"beta-{index}" for index in range(4)],
            "ground_truth": labels,
            "predictions": [0, 0, 1, 1],
        },
    }


def _attach_evidence(node: Node, root) -> Node:
    evidence_dir = root / node.id
    evidence_dir.mkdir(parents=True)
    artifact = evidence_dir / "experiment_data.npy"
    np.save(artifact, _payload())
    report = evaluate_experiment_data(artifact, requested_metric="accuracy")
    assert report["status"] == "verified"
    node.metric = MetricValue(report["metric"])
    node.metric_provenance = "deterministic_verified"
    node.evaluation_report = report
    node.exp_results_dir = str(evidence_dir.resolve())
    node.is_buggy = False
    node.is_buggy_plots = False
    return node


def _journal(tmp_path, stage_index: int = 1) -> Journal:
    stage_name = f"{stage_index}_qualified_stage"
    root = tmp_path / stage_name
    qualified = _attach_evidence(
        Node(
            code=CONTROL_CODE,
            id=f"qualified-{stage_index}",
            plan="Compare the locked method with its control",
            analysis="The verified trend is stable across datasets",
        ),
        root,
    )
    plot = root / qualified.id / "trend.png"
    plot.write_bytes(b"plot-evidence")
    qualified.plot_paths = [str(plot.resolve())]
    qualified.plot_analyses = [
        {
            "plot_path": str(plot.resolve()),
            "analysis": "Observed stable trend",
        }
    ]

    seeds = []
    for seed in CONFIRMATION_SEEDS:
        seeded_code, bootstrap_hash = _inject_seed_bootstrap(qualified.code, seed)
        seed_node = _attach_evidence(
            Node(
                code=seeded_code,
                id=f"seed-{stage_index}-{seed}",
                parent=qualified,
                is_seed_node=True,
                random_seed=seed,
                seed_bootstrap_hash=bootstrap_hash,
                plan="Repeat the locked experiment",
                analysis="The confirmation run follows the locked method",
            ),
            root,
        )
        seeds.append(seed_node)

    qualified.multi_seed_report = _build_multi_seed_report(
        stage=SimpleNamespace(name=stage_name),
        node=qualified,
        seed_nodes=seeds,
        configured_seeds=CONFIRMATION_SEEDS,
        max_relative_ci_half_width=0.25,
        absolute_ci_floor=0.01,
        control_report=None,
    )
    return Journal(nodes=[qualified, *seeds])


def _advisory(**updates) -> dict:
    value = {
        "schema": log_summarization.REPORT_ADVISORY_SCHEMA,
        "Experiment_description": [
            "experiment.qualified_method_replay",
        ],
        "Significance": ["significance.internal_consistency_only"],
        "Description": ["description.agent_interpretation_advisory"],
        "List_of_included_plots": [],
        "Key_numerical_results": [],
    }
    value.update(updates)
    return value


def _model_response(advisory: dict) -> tuple[str, list]:
    return f"```json\n{json.dumps(advisory)}\n```", []


def test_stage_summary_deterministically_renders_bound_numerical_results(
    tmp_path,
) -> None:
    journal = _journal(tmp_path)
    evidence = log_summarization._validated_stage_evidence(journal)
    plot_claim_id = evidence["plot_manifest"]["entries"][0]["plot_claim_id"]
    with mock.patch.object(
        log_summarization,
        "get_response_from_llm",
        return_value=_model_response(_advisory(List_of_included_plots=[plot_claim_id])),
    ):
        summary = log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )

    assert summary["schema"] == log_summarization.REPORT_SUMMARY_SCHEMA
    assert summary["stage_name"] == "1_qualified_stage"
    assert summary["List_of_included_plots"] == [
        {
            "plot_claim_id": plot_claim_id,
            "caption": log_summarization._PLOT_CAPTION,
        }
    ]
    manifest = summary["Plot_evidence_manifest"]
    assert manifest["schema"] == log_summarization.PLOT_MANIFEST_SCHEMA
    assert manifest["path_base"] == "qualified.exp_results_dir"
    assert manifest["artifact_binding_scope"] == (
        "artifact_identity_only_not_scientific_verification"
    )
    assert manifest["manifest_hash"].startswith("sha256:")
    plot_entry = manifest["entries"][0]
    assert plot_entry["path"] == "trend.png"
    assert not plot_entry["path"].startswith("/")
    assert plot_entry["content_sha256"].startswith("sha256:")
    assert plot_entry["qualified_node_id"] == journal.nodes[0].id
    assert (
        plot_entry["evaluation_result_hash"]
        == journal.nodes[0].evaluation_report["result_hash"]
    )
    assert (
        plot_entry["multi_seed_receipt_hash"]
        == journal.nodes[0].multi_seed_report["receipt_hash"]
    )
    results = summary["Key_numerical_results"]
    assert [row["dataset_name"] for row in results] == [
        "dataset_alpha",
        "dataset_beta",
    ]
    assert results[0]["confirmation_mean"] == pytest.approx(0.75)
    assert results[0]["ci95_lower"] == pytest.approx(0.75)
    assert results[0]["ci95_upper"] == pytest.approx(0.75)
    assert results[0]["n"] == 3
    qualified = journal.nodes[0]
    for row in results:
        assert row["qualified_node_id"] == qualified.id
        assert (
            row["evaluation_result_hash"] == qualified.evaluation_report["result_hash"]
        )
        assert (
            row["multi_seed_receipt_hash"]
            == qualified.multi_seed_report["receipt_hash"]
        )
        assert row["verification_scope"] == "artifact_internal_consistency"


@pytest.mark.parametrize(
    "bad_advisory, message",
    [
        (
            _advisory(
                Key_numerical_results=[
                    {"result": 0.99, "description": "invented", "analysis": "invented"}
                ]
            ),
            "must not provide numerical",
        ),
        ({**_advisory(), "unexpected": "field"}, "advisory schema"),
        (
            _advisory(Experiment_description=["ninety nine percent"]),
            "invalid qualitative claim identifier",
        ),
        (
            _advisory(List_of_included_plots=["plot-claim:forged-relative-path"]),
            "not bound to allowed evidence",
        ),
        (
            _advisory(Description="The curve improves by forty two percent"),
            "select qualitative claim identifiers",
        ),
    ],
)
def test_stage_summary_rejects_untrusted_model_claims(
    tmp_path,
    bad_advisory,
    message,
) -> None:
    journal = _journal(tmp_path)
    with (
        mock.patch.object(
            log_summarization,
            "get_response_from_llm",
            return_value=_model_response(bad_advisory),
        ),
        pytest.raises(ValueError, match=message),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )


def test_plot_reference_is_resolved_relative_to_evidence_root(tmp_path) -> None:
    journal = _journal(tmp_path)
    qualified = journal.nodes[0]
    qualified.plot_paths = ["trend.png"]
    qualified.plot_analyses[0]["plot_path"] = "trend.png"
    evidence = log_summarization._validated_stage_evidence(journal)
    claim_id = evidence["plot_manifest"]["entries"][0]["plot_claim_id"]

    with mock.patch.object(
        log_summarization,
        "get_response_from_llm",
        return_value=_model_response(_advisory(List_of_included_plots=[claim_id])),
    ):
        summary = log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )

    assert summary["Plot_evidence_manifest"]["entries"][0]["path"] == "trend.png"


def test_plot_reference_rejects_forged_relative_escape(tmp_path) -> None:
    journal = _journal(tmp_path)
    qualified = journal.nodes[0]
    forged = Path(qualified.exp_results_dir).parent / "forged.png"
    forged.write_bytes(b"forged-outside-qualified-root")
    qualified.plot_paths = ["../forged.png"]
    qualified.plot_analyses[0]["plot_path"] = "../forged.png"

    with (
        mock.patch.object(log_summarization, "get_response_from_llm") as model_call,
        pytest.raises(ValueError, match="escapes its evidence root"),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )
    model_call.assert_not_called()


def test_plot_reference_rejects_absolute_file_outside_evidence_root(tmp_path) -> None:
    journal = _journal(tmp_path)
    qualified = journal.nodes[0]
    forged = tmp_path / "absolute-forged.png"
    forged.write_bytes(b"outside-qualified-root")
    qualified.plot_paths = [str(forged.resolve())]
    qualified.plot_analyses[0]["plot_path"] = str(forged.resolve())

    with (
        mock.patch.object(log_summarization, "get_response_from_llm") as model_call,
        pytest.raises(ValueError, match="escapes its evidence root"),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )
    model_call.assert_not_called()


def test_plot_reference_rejects_symlink(tmp_path) -> None:
    journal = _journal(tmp_path)
    qualified = journal.nodes[0]
    plot = Path(qualified.plot_paths[0])
    target = plot.with_name("target.png")
    target.write_bytes(b"regular-target")
    plot.unlink()
    plot.symlink_to(target.name)

    with (
        mock.patch.object(log_summarization, "get_response_from_llm") as model_call,
        pytest.raises(ValueError, match="must not use symlinks"),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )
    model_call.assert_not_called()


def test_plot_reference_rejects_missing_file(tmp_path) -> None:
    journal = _journal(tmp_path)
    Path(journal.nodes[0].plot_paths[0]).unlink()

    with (
        mock.patch.object(log_summarization, "get_response_from_llm") as model_call,
        pytest.raises(ValueError, match="file is unavailable"),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )
    model_call.assert_not_called()


def test_plot_modified_during_model_call_invalidates_manifest(tmp_path) -> None:
    journal = _journal(tmp_path)
    evidence = log_summarization._validated_stage_evidence(journal)
    claim_id = evidence["plot_manifest"]["entries"][0]["plot_claim_id"]
    plot = Path(journal.nodes[0].plot_paths[0])

    def mutate_plot(*_args, **_kwargs):
        plot.write_bytes(b"modified-after-model-context")
        return _model_response(_advisory(List_of_included_plots=[claim_id]))

    with (
        mock.patch.object(
            log_summarization,
            "get_response_from_llm",
            side_effect=mutate_plot,
        ),
        pytest.raises(ValueError, match="plot manifest is not artifact-bound"),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )


def test_completed_summary_rejects_rehashed_forged_plot_content_hash(
    tmp_path,
) -> None:
    journal = _journal(tmp_path)
    with mock.patch.object(
        log_summarization,
        "get_response_from_llm",
        return_value=_model_response(_advisory()),
    ):
        summary = log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )

    forged = copy.deepcopy(summary)
    manifest = forged["Plot_evidence_manifest"]
    manifest["entries"][0]["content_sha256"] = "sha256:" + "0" * 64
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = log_summarization._strict_sha256_json(unsigned)

    with pytest.raises(ValueError, match="plot manifest is not artifact-bound"):
        log_summarization._validate_completed_summary(
            forged,
            stage_name="1_qualified_stage",
            journal=journal,
        )


@pytest.mark.parametrize(
    "corruption", ["evaluation_hash", "receipt_hash", "nan", "missing"]
)
def test_stage_summary_fails_closed_on_bad_evidence(tmp_path, corruption) -> None:
    journal = _journal(tmp_path)
    qualified = journal.nodes[0]
    if corruption == "evaluation_hash":
        qualified.evaluation_report["result_hash"] = "sha256:" + "0" * 64
    elif corruption == "receipt_hash":
        qualified.multi_seed_report["receipt_hash"] = "sha256:" + "0" * 64
    elif corruption == "nan":
        qualified.multi_seed_report["datasets"]["dataset_alpha"]["mean"] = float("nan")
    else:
        qualified.multi_seed_report.pop("receipt_hash")

    with (
        mock.patch.object(
            log_summarization,
            "get_response_from_llm",
            return_value=_model_response(_advisory()),
        ) as model_call,
        pytest.raises(ValueError),
    ):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )
    model_call.assert_not_called()


def test_stage_summary_rejects_finite_stats_rewritten_with_a_new_receipt(
    tmp_path,
) -> None:
    journal = _journal(tmp_path)
    report = journal.nodes[0].multi_seed_report
    report["datasets"]["dataset_alpha"]["mean"] = 0.99
    unsigned = copy.deepcopy(report)
    unsigned.pop("receipt_hash")
    report["receipt_hash"] = _strict_sha256_json(unsigned)

    with pytest.raises(ValueError, match="statistics"):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )


def test_final_summary_rejects_non_qualified_stage_input(tmp_path) -> None:
    journal = _journal(tmp_path)
    extra = _attach_evidence(
        Node(
            code=CONTROL_CODE,
            id="exploratory",
            plan="Unselected exploration",
            analysis="This node was not promoted by the gate",
        ),
        tmp_path / "exploratory-evidence",
    )
    journal.append(extra)

    with pytest.raises(ValueError, match="non-qualified"):
        log_summarization.get_stage_summary(
            journal,
            "1_qualified_stage",
            "report/glm-5.3",
            object(),
        )


def test_stage_summary_rejects_a_relabelled_receipt_before_model_call(
    tmp_path,
) -> None:
    journal = _journal(tmp_path)

    with (
        mock.patch.object(log_summarization, "get_response_from_llm") as model_call,
        pytest.raises(ValueError, match="stage name does not match"),
    ):
        log_summarization.get_stage_summary(
            journal,
            "2_relabelled_stage",
            "report/glm-5.3",
            object(),
        )

    model_call.assert_not_called()


@pytest.mark.parametrize("relabel", [False, True])
def test_overall_summary_rejects_reordered_stage_receipts(
    tmp_path,
    relabel,
) -> None:
    cfg = SimpleNamespace(report=SimpleNamespace(model="report/model", temp=0.2))
    journals = [
        (f"{index}_qualified_stage", _journal(tmp_path, index)) for index in range(1, 5)
    ]
    journals[0], journals[1] = journals[1], journals[0]
    submitted = (
        [
            (f"{index}_qualified_stage", journal)
            for index, (_old_name, journal) in enumerate(journals, start=1)
        ]
        if relabel
        else journals
    )
    expected_error = "stage name does not match" if relabel else "canonical order"

    with (
        mock.patch.object(log_summarization, "get_ai_client", return_value=object()),
        mock.patch.object(log_summarization, "get_response_from_llm") as model_call,
        pytest.raises(ValueError, match=expected_error),
    ):
        log_summarization.overall_summarize(submitted, cfg)

    model_call.assert_not_called()


def test_final_summary_routes_all_narratives_through_report_model_only(
    tmp_path,
) -> None:
    cfg = SimpleNamespace(
        report=SimpleNamespace(model="report/glm-5.3", temp=0.2),
        agent=SimpleNamespace(
            summary=SimpleNamespace(model="wrong/summary-model", temp=1.0)
        ),
    )
    journals = [
        (f"{index}_qualified_stage", _journal(tmp_path, index)) for index in range(1, 5)
    ]

    with (
        mock.patch.object(
            log_summarization,
            "get_ai_client",
            return_value=object(),
        ) as get_client,
        mock.patch.object(
            log_summarization,
            "get_response_from_llm",
            return_value=_model_response(_advisory()),
        ) as summarize,
    ):
        result = log_summarization.overall_summarize(journals, cfg)

    get_client.assert_called_once_with("report/glm-5.3")
    assert summarize.call_count == 4
    assert all(call.args[2] == "report/glm-5.3" for call in summarize.call_args_list)
    assert len(result) == 4
    assert all(
        item["schema"] == log_summarization.REPORT_SUMMARY_SCHEMA for item in result
    )


@pytest.mark.parametrize("untrusted", [None, {}, {"summary": "qualified"}])
def test_overall_summary_rejects_arbitrary_helper_output(tmp_path, untrusted) -> None:
    cfg = SimpleNamespace(report=SimpleNamespace(model="report/model", temp=0.2))
    journals = [
        (f"{index}_qualified_stage", _journal(tmp_path, index)) for index in range(1, 5)
    ]

    with (
        mock.patch.object(log_summarization, "get_ai_client", return_value=object()),
        mock.patch.object(
            log_summarization,
            "get_stage_summary",
            return_value=untrusted,
        ),
        pytest.raises(ValueError, match="evidence-bound schema"),
    ):
        log_summarization.overall_summarize(journals, cfg)
