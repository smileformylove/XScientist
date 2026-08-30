from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ai_scientist.utils.authority_attempts import (
    AuthorityAttemptError,
    begin_authority_attempt,
    canonical_authority_hash,
    inspect_authority_attempts,
    persist_authority_object,
    record_authority_attempt_result,
    run_authority_call,
)


def _begin(log_dir: Path) -> str:
    spec_hash, spec_ref = persist_authority_object(
        log_dir,
        category="implementation-spec",
        payload={"schema": "test.spec.v1", "objective": "bounded"},
    )
    attempt_id = begin_authority_attempt(
        log_dir,
        spec_hash=spec_hash,
        spec_ref=spec_ref,
        parent_node_id="parent-1",
        role="implementation",
        model="openai_compat/glm-5.3",
        task_kind="baseline",
    )
    assert attempt_id is not None
    return attempt_id


def test_planned_attempt_is_append_only_and_replayable(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    attempt_id = _begin(log_dir)
    planned_path = log_dir / "authority_attempts" / attempt_id / "0.json"
    planned_before = planned_path.read_bytes()

    audit = inspect_authority_attempts(log_dir)
    assert audit["valid"] is False
    assert audit["errors"] == ["attempt_incomplete"]
    assert audit["incomplete_attempt_ids"] == [attempt_id]
    assert audit["orphan_attempt_ids"] == []

    record_authority_attempt_result(
        log_dir,
        attempt_id,
        status="accepted",
        result_ref="sha256:" + "a" * 64,
    )

    assert planned_path.read_bytes() == planned_before
    events = sorted((planned_path.parent).glob("*.json"))
    assert [path.name for path in events] == ["0.json", "1.json"]
    result = json.loads(events[1].read_text(encoding="utf-8"))
    assert (
        result["previous_event_hash"]
        == json.loads(planned_before.decode("utf-8"))["event_hash"]
    )
    assert result["result_hash"].startswith("sha256:")
    assert result["result_ref"].startswith("authority_objects/attempt-result/")
    result_object = json.loads(
        (log_dir / result["result_ref"]).read_text(encoding="utf-8")
    )
    assert result_object == {
        "declared_content_hash": "sha256:" + "a" * 64,
        "result": None,
        "schema": "xscientist.authority-attempt-result.v1",
    }
    audit = inspect_authority_attempts(log_dir)
    assert audit["valid"] is True
    assert audit["incomplete_attempt_ids"] == []
    assert audit["attempts"][0]["status"] == "accepted"


def test_attempt_replay_reports_orphan_and_tamper(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    attempt_id = _begin(log_dir)
    orphan_id = "attempt-" + "f" * 32
    orphan = log_dir / "authority_attempts" / orphan_id
    orphan.mkdir(mode=0o700)

    planned_path = log_dir / "authority_attempts" / attempt_id / "0.json"
    payload = json.loads(planned_path.read_text(encoding="utf-8"))
    payload["model"] = "tampered/model"
    planned_path.write_text(json.dumps(payload), encoding="utf-8")

    audit = inspect_authority_attempts(log_dir)
    assert audit["valid"] is False
    assert orphan_id in audit["orphan_attempt_ids"]
    assert attempt_id in audit["invalid_attempt_ids"]


def test_attempt_replay_rejects_tampered_spec_object(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    attempt_id = _begin(log_dir)
    planned = json.loads(
        (log_dir / "authority_attempts" / attempt_id / "0.json").read_text(
            encoding="utf-8"
        )
    )
    spec_path = log_dir / planned["spec_ref"]
    spec_path.write_text('{"objective":"changed"}', encoding="utf-8")

    audit = inspect_authority_attempts(log_dir)
    assert audit["valid"] is False
    assert audit["invalid_attempt_ids"] == [attempt_id]


def test_terminal_result_never_overwrites_or_forks(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    attempt_id = _begin(log_dir)
    first_hash = record_authority_attempt_result(
        log_dir,
        attempt_id,
        status="rejected",
        result_ref="sha256:" + "b" * 64,
    )
    assert (
        record_authority_attempt_result(
            log_dir,
            attempt_id,
            status="rejected",
            result_ref="sha256:" + "b" * 64,
        )
        == first_hash
    )
    with pytest.raises(AuthorityAttemptError, match="already complete"):
        record_authority_attempt_result(
            log_dir,
            attempt_id,
            status="rejected",
            result_ref="sha256:" + "c" * 64,
        )
    with pytest.raises(AuthorityAttemptError, match="already complete"):
        record_authority_attempt_result(
            log_dir,
            attempt_id,
            status="rejected",
            result_ref="sha256:" + "b" * 64,
            error_type="DifferentTerminalMeaning",
        )
    assert len(list((log_dir / "authority_attempts" / attempt_id).iterdir())) == 2


def test_failed_authority_call_has_a_terminal_receipt(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    def fail() -> object:
        raise RuntimeError("private provider response must not be persisted")

    with pytest.raises(RuntimeError, match="private provider"):
        run_authority_call(
            log_dir,
            category="selector-spec",
            specification={"candidate_ids": ["one", "two"]},
            parent_node_id=None,
            role="select_node",
            model="openai/judgment",
            task_kind="node_selection",
            operation=fail,
        )

    audit = inspect_authority_attempts(log_dir)
    assert audit["valid"] is True
    assert audit["attempts"][0]["status"] == "failed"
    result = json.loads(
        (
            log_dir
            / "authority_attempts"
            / audit["attempts"][0]["attempt_id"]
            / "1.json"
        ).read_text(encoding="utf-8")
    )
    assert result["error_type"] == "RuntimeError"
    assert "private provider response" not in json.dumps(result)


def test_attempt_root_symlink_is_rejected(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (log_dir / "authority_attempts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(AuthorityAttemptError, match="root is unsafe"):
        _begin(log_dir)


def test_missing_attempt_root_is_not_created_or_reported_valid(tmp_path: Path) -> None:
    log_dir = tmp_path / "missing-logs"

    audit = inspect_authority_attempts(log_dir)

    assert audit["valid"] is False
    assert audit["errors"] == ["attempt_root_missing"]
    assert not log_dir.exists()


def test_authority_hash_rejects_non_json_and_nan() -> None:
    with pytest.raises(AuthorityAttemptError, match="strict JSON"):
        canonical_authority_hash({"unsupported": object()})
    with pytest.raises(AuthorityAttemptError, match="strict JSON"):
        canonical_authority_hash({"value": float("nan")})


def test_replay_rejects_duplicate_keys_and_noncanonical_json(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    duplicate_attempt = _begin(log_dir)
    duplicate_path = log_dir / "authority_attempts" / duplicate_attempt / "0.json"
    duplicate_payload = duplicate_path.read_bytes()
    duplicate_path.write_bytes(duplicate_payload[:-1] + b',"status":"planned"}')

    noncanonical_attempt = _begin(log_dir)
    noncanonical_path = log_dir / "authority_attempts" / noncanonical_attempt / "0.json"
    parsed = json.loads(noncanonical_path.read_text(encoding="utf-8"))
    noncanonical_path.write_text(
        json.dumps(parsed, ensure_ascii=False, sort_keys=False, indent=2),
        encoding="utf-8",
    )

    audit = inspect_authority_attempts(log_dir)

    errors = {row["attempt_id"]: row["errors"] for row in audit["attempts"]}
    assert "duplicate keys" in errors[duplicate_attempt][0]
    assert "not canonical JSON" in errors[noncanonical_attempt][0]


def test_expected_attempt_ids_fail_closed_without_creating_state(
    tmp_path: Path,
) -> None:
    missing_id = "attempt-" + "e" * 32
    audit = inspect_authority_attempts(
        tmp_path / "logs",
        expected_attempt_ids=[missing_id],
    )

    assert audit["expected_valid"] is False
    assert audit["missing_expected_attempt_ids"] == [missing_id]
    assert not (tmp_path / "logs").exists()


def test_relative_log_dir_persists_under_current_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    attempt_id = _begin(Path("relative-logs"))

    assert (
        tmp_path / "relative-logs" / "authority_attempts" / attempt_id / "0.json"
    ).is_file()


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership contract")
def test_attempt_directories_are_private(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    attempt_id = _begin(log_dir)

    root_mode = stat_mode(log_dir / "authority_attempts")
    attempt_mode = stat_mode(log_dir / "authority_attempts" / attempt_id)
    assert root_mode & 0o077 == 0
    assert attempt_mode & 0o077 == 0


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777
