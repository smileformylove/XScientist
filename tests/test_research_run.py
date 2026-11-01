from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from jsonschema import validate

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.schemas import load_schema
from xscientist.research_actions import ResearchActionLedger
from xscientist.research_cli import main
from xscientist.research_git import ResearchGitError
from xscientist.research_run import (
    ResearchRunLedger,
    list_research_runs,
    summarize_research_run,
)
from xscientist.research_vcs import ResearchRepository


pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="Git is required")


def _repo(tmp_path: Path) -> ResearchRepository:
    return ResearchRepository.init(
        tmp_path / "study",
        question="Can a bounded run resume without copying private context?",
        git_user_name="Run Test",
        git_user_email="run@example.invalid",
    )


def test_run_steps_are_hash_bound_and_context_only(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    action = ResearchActionLedger(repo).begin(
        "act-video-inspect",
        "video.inspect",
        {"prompt": "private prompt", "asset": "rso-deadbeefdeadbeef"},
    )
    context = repo.record("evidence", {"summary": "existing video cue"})
    ledger = ResearchRunLedger(repo)
    step = ledger.append(
        "run-video-1",
        "inspect-1",
        "inspect",
        "running",
        {"prompt": "private prompt"},
        action_ref=action["object_id"],
        context_refs=[context.object_id],
        next_action="video.inspect",
    )
    assert step["payload"]["input_hash"] == canonical_content_hash(
        {"prompt": "private prompt"}
    )
    assert "private prompt" not in str(step)
    assert step["payload"]["context_refs"] == [context.object_id]
    validate(step["payload"], load_schema("research_run"))


def test_sequence_is_monotonic_and_idempotent(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    ledger = ResearchRunLedger(repo)
    first = ledger.append("run-1", "plan", "plan", "completed", {"question": "q"})
    second = ledger.append(
        "run-1", "inspect", "inspect", "waiting", {"asset": "a"}, requires_human=True
    )
    repeated = ledger.append(
        "run-1", "inspect", "inspect", "waiting", {"asset": "a"},
        requires_human=True, sequence=1
    )
    assert repeated["object_id"] == second["object_id"]
    with pytest.raises(ResearchGitError, match="advance by one"):
        ledger.append("run-1", "gap", "execute", "running", {"x": 1}, sequence=3)
    assert [row["payload"]["sequence"] for row in ledger.history("run-1")] == [0, 1]
    assert first["payload"]["sequence"] == 0


def test_summary_surfaces_human_attention_before_history(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    ledger = ResearchRunLedger(repo)
    ledger.append("run-2", "plan", "plan", "completed", {"question": "q"})
    ledger.append("run-2", "inspect", "inspect", "blocked", {"asset": "a"}, next_action="video.inspect")
    summary = summarize_research_run(repo, "run-2")
    assert summary["attention_required"] is True
    assert summary["attention"][0]["step_id"] == "inspect"
    assert summary["attention"][0]["next_action"] == "video.inspect"
    assert summary["raw_content_included"] is False
    assert summary["summary_hash"].startswith("sha256:")
    assert list_research_runs(repo)[0]["run_id"] == "run-2"


def test_references_must_exist(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    with pytest.raises(ResearchGitError, match="action_ref"):
        ResearchRunLedger(repo).append(
            "run-3", "execute", "execute", "running", {"x": 1}, action_ref="rso-0000000000000000"
        )


def test_cli_show_json_returns_after_rendering(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repo = _repo(tmp_path)
    ledger = ResearchRunLedger(repo)
    ledger.append("run-cli", "plan", "plan", "running", {"question": "q"})

    assert main(["run", "show", "run-cli", "--repo", str(repo.path), "--json"]) == 0
    output = capsys.readouterr().out
    assert '"run_id": "run-cli"' in output
