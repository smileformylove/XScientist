from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from xscientist import ResearchRepository
from xscientist import research_git
from xscientist.research_git import (
    ResearchGitError,
    _claim_experiment_admission_start,
    _experiment_admission_terminal_reserved,
    _load_experiment_admission_execution_claim,
    _load_experiment_admission_terminal_reservation,
    _repository_common_git_dir,
    _reserve_experiment_admission_terminal,
)

START_ID = "rso-1111111111111111"
ATTEMPT_ID = "rso-2222222222222222"
OTHER_ATTEMPT_ID = "rso-3333333333333333"
RECEIPT_HASH = "sha256:" + "4" * 64
OTHER_RECEIPT_HASH = "sha256:" + "5" * 64
ATTEMPT_HASH = "sha256:" + "6" * 64
OTHER_ATTEMPT_HASH = "sha256:" + "7" * 64


@pytest.fixture
def repository(tmp_path: Path) -> ResearchRepository:
    return ResearchRepository.init(
        tmp_path / "research",
        question="Are admission runtime markers durable?",
        git_user_name="Admission Storage Test",
        git_user_email="admission-storage@example.invalid",
    )


def _state_directory(repository: ResearchRepository, name: str) -> Path:
    return _repository_common_git_dir(repository.path) / "xscientist" / name


def _reserve(repository: ResearchRepository, **overrides: str) -> Path:
    values = {
        "start_object_id": START_ID,
        "receipt_hash": RECEIPT_HASH,
        "attempt_object_id": ATTEMPT_ID,
        "attempt_object_hash": ATTEMPT_HASH,
        **overrides,
    }
    return _reserve_experiment_admission_terminal(repository.path, **values)


def test_execution_claim_reader_distinguishes_missing_valid_and_mismatch(
    repository: ResearchRepository,
) -> None:
    assert (
        _load_experiment_admission_execution_claim(
            repository.path, start_object_id=START_ID
        )
        is None
    )

    path = _claim_experiment_admission_start(
        repository.path,
        start_object_id=START_ID,
        receipt_hash=RECEIPT_HASH,
    )
    claim = _load_experiment_admission_execution_claim(
        repository.path,
        start_object_id=START_ID,
        receipt_hash=RECEIPT_HASH,
    )

    assert claim is not None
    assert claim["start_object_id"] == START_ID
    assert claim["receipt_hash"] == RECEIPT_HASH
    assert claim["process_id"] == os.getpid()
    assert path.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(ResearchGitError, match="receipt hash does not match"):
        _load_experiment_admission_execution_claim(
            repository.path,
            start_object_id=START_ID,
            receipt_hash=OTHER_RECEIPT_HASH,
        )
    with pytest.raises(ResearchGitError, match="already been claimed for execution"):
        _claim_experiment_admission_start(
            repository.path,
            start_object_id=START_ID,
            receipt_hash=RECEIPT_HASH,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("receipt_hash", "not-a-hash", "marker is invalid"),
        ("claimed_at", "not-a-timestamp", "timestamp is invalid"),
        ("process_id", True, "process id is invalid"),
        ("unexpected", "field", "marker is invalid"),
    ],
)
def test_execution_claim_reader_rejects_invalid_existing_marker(
    repository: ResearchRepository,
    field: str,
    value: object,
    message: str,
) -> None:
    path = _claim_experiment_admission_start(
        repository.path,
        start_object_id=START_ID,
        receipt_hash=RECEIPT_HASH,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchGitError, match=message):
        _load_experiment_admission_execution_claim(
            repository.path, start_object_id=START_ID
        )


def test_execution_claim_reader_rejects_symlink_marker(
    repository: ResearchRepository, tmp_path: Path
) -> None:
    claims = _state_directory(repository, "admission-claims")
    claims.mkdir(parents=True)
    external = tmp_path / "external-claim.json"
    external.write_text("{}\n", encoding="utf-8")
    (claims / f"{START_ID}.json").symlink_to(external)

    with pytest.raises(ResearchGitError, match="marker is unsafe"):
        _load_experiment_admission_execution_claim(
            repository.path, start_object_id=START_ID
        )


def test_execution_claim_partial_write_never_creates_final_marker(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = os.write
    write_count = 0

    def partial_then_fail(descriptor: int, data: bytes) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            return real_write(descriptor, data[: max(1, len(data) // 2)])
        raise OSError("simulated claim write failure")

    monkeypatch.setattr(research_git.os, "write", partial_then_fail)
    with pytest.raises(OSError, match="simulated claim write failure"):
        _claim_experiment_admission_start(
            repository.path,
            start_object_id=START_ID,
            receipt_hash=RECEIPT_HASH,
        )

    assert (
        _load_experiment_admission_execution_claim(
            repository.path, start_object_id=START_ID
        )
        is None
    )
    assert list(_state_directory(repository, "admission-claims").glob("*.tmp")) == []


def test_execution_claim_keeps_complete_marker_after_directory_fsync_uncertainty(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    _state_directory(repository, "admission-claims").mkdir(parents=True)

    def uncertain(_path: Path) -> None:
        raise OSError("simulated directory fsync uncertainty")

    monkeypatch.setattr(research_git, "_fsync_directory", uncertain)
    with pytest.raises(OSError, match="directory fsync uncertainty"):
        _claim_experiment_admission_start(
            repository.path,
            start_object_id=START_ID,
            receipt_hash=RECEIPT_HASH,
        )

    claim = _load_experiment_admission_execution_claim(
        repository.path,
        start_object_id=START_ID,
        receipt_hash=RECEIPT_HASH,
    )
    assert claim is not None
    with pytest.raises(ResearchGitError, match="already been claimed for execution"):
        _claim_experiment_admission_start(
            repository.path,
            start_object_id=START_ID,
            receipt_hash=RECEIPT_HASH,
        )


def test_execution_claim_link_race_remains_single_use(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    _state_directory(repository, "admission-claims").mkdir(parents=True)
    real_link = os.link

    def publish_competing_claim(source: str, destination: str) -> None:
        real_link(source, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(research_git.os, "link", publish_competing_claim)
    with pytest.raises(ResearchGitError, match="already been claimed for execution"):
        _claim_experiment_admission_start(
            repository.path,
            start_object_id=START_ID,
            receipt_hash=RECEIPT_HASH,
        )

    claim = _load_experiment_admission_execution_claim(
        repository.path,
        start_object_id=START_ID,
        receipt_hash=RECEIPT_HASH,
    )
    assert claim is not None


def test_terminal_reservation_reader_and_exact_retry_are_idempotent(
    repository: ResearchRepository,
) -> None:
    assert (
        _load_experiment_admission_terminal_reservation(
            repository.path, start_object_id=START_ID
        )
        is None
    )
    first = _reserve(repository)
    original_bytes = first.read_bytes()

    second = _reserve(repository)
    reservation = _load_experiment_admission_terminal_reservation(
        repository.path, start_object_id=START_ID
    )

    assert second == first
    assert second.read_bytes() == original_bytes
    assert reservation is not None
    assert reservation["attempt_object_id"] == ATTEMPT_ID
    assert reservation["attempt_object_hash"] == ATTEMPT_HASH
    assert _experiment_admission_terminal_reserved(
        repository.path, start_object_id=START_ID
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "wrong-schema", "marker is invalid"),
        ("start_object_id", OTHER_ATTEMPT_ID, "marker is invalid"),
        ("reserved_at", "2026-09-05T00:00:00", "timestamp is invalid"),
        ("process_id", True, "process id is invalid"),
        ("unexpected", "field", "marker is invalid"),
    ],
)
def test_terminal_reservation_reader_rejects_invalid_existing_marker(
    repository: ResearchRepository,
    field: str,
    value: object,
    message: str,
) -> None:
    path = _reserve(repository)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ResearchGitError, match=message):
        _load_experiment_admission_terminal_reservation(
            repository.path, start_object_id=START_ID
        )


def test_terminal_reservation_reader_rejects_symlink_marker(
    repository: ResearchRepository, tmp_path: Path
) -> None:
    reservations = _state_directory(repository, "admission-terminal-reservations")
    reservations.mkdir(parents=True)
    external = tmp_path / "external-reservation.json"
    external.write_text("{}\n", encoding="utf-8")
    (reservations / f"{START_ID}.json").symlink_to(external)

    with pytest.raises(ResearchGitError, match="marker is unsafe"):
        _load_experiment_admission_terminal_reservation(
            repository.path, start_object_id=START_ID
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"receipt_hash": OTHER_RECEIPT_HASH},
        {"attempt_object_id": OTHER_ATTEMPT_ID},
        {"attempt_object_hash": OTHER_ATTEMPT_HASH},
    ],
)
def test_terminal_reservation_conflicting_identity_fails_closed(
    repository: ResearchRepository, overrides: dict[str, str]
) -> None:
    path = _reserve(repository)
    original_bytes = path.read_bytes()

    with pytest.raises(ResearchGitError, match="conflicting attempt identity"):
        _reserve(repository, **overrides)

    assert path.read_bytes() == original_bytes


def test_terminal_reservation_link_race_with_exact_identity_is_idempotent(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservations = _state_directory(repository, "admission-terminal-reservations")
    reservations.mkdir(parents=True)
    real_link = os.link

    def publish_competing_reservation(source: str, destination: str) -> None:
        real_link(source, destination)
        raise FileExistsError(destination)

    monkeypatch.setattr(research_git.os, "link", publish_competing_reservation)
    path = _reserve(repository)

    assert path == reservations / f"{START_ID}.json"
    reservation = _load_experiment_admission_terminal_reservation(
        repository.path, start_object_id=START_ID
    )
    assert reservation is not None
    assert reservation["attempt_object_hash"] == ATTEMPT_HASH


def test_terminal_reservation_link_race_with_conflicting_identity_fails_closed(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservations = _state_directory(repository, "admission-terminal-reservations")
    reservations.mkdir(parents=True)
    real_link = os.link

    def publish_conflicting_reservation(source: str, destination: str) -> None:
        payload = json.loads(Path(source).read_text(encoding="utf-8"))
        payload["attempt_object_hash"] = OTHER_ATTEMPT_HASH
        competing = Path(source).with_name(".competing-terminal.tmp")
        competing.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        try:
            real_link(competing, destination)
        finally:
            competing.unlink(missing_ok=True)
        raise FileExistsError(destination)

    monkeypatch.setattr(research_git.os, "link", publish_conflicting_reservation)
    with pytest.raises(ResearchGitError, match="conflicting attempt identity"):
        _reserve(repository)

    reservation = _load_experiment_admission_terminal_reservation(
        repository.path, start_object_id=START_ID
    )
    assert reservation is not None
    assert reservation["attempt_object_hash"] == OTHER_ATTEMPT_HASH


def test_terminal_reservation_reader_rejects_truncated_marker(
    repository: ResearchRepository,
) -> None:
    path = _reserve(repository)
    path.write_bytes(b'{"schema":')

    with pytest.raises(ResearchGitError, match="invalid JSON"):
        _load_experiment_admission_terminal_reservation(
            repository.path, start_object_id=START_ID
        )
    with pytest.raises(ResearchGitError, match="invalid JSON"):
        _experiment_admission_terminal_reserved(
            repository.path, start_object_id=START_ID
        )
    with pytest.raises(ResearchGitError, match="invalid JSON"):
        _reserve(repository)


def test_terminal_reservation_partial_write_never_creates_final_marker(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_write = os.write
    write_count = 0

    def partial_then_fail(descriptor: int, data: bytes) -> int:
        nonlocal write_count
        write_count += 1
        if write_count == 1:
            return real_write(descriptor, data[: max(1, len(data) // 2)])
        raise OSError("simulated reservation write failure")

    monkeypatch.setattr(research_git.os, "write", partial_then_fail)
    with pytest.raises(OSError, match="simulated reservation write failure"):
        _reserve(repository)

    assert (
        _load_experiment_admission_terminal_reservation(
            repository.path, start_object_id=START_ID
        )
        is None
    )
    assert (
        list(
            _state_directory(repository, "admission-terminal-reservations").glob(
                "*.tmp"
            )
        )
        == []
    )


def test_terminal_reservation_exact_retry_recovers_directory_fsync_uncertainty(
    repository: ResearchRepository, monkeypatch: pytest.MonkeyPatch
) -> None:
    reservations = _state_directory(repository, "admission-terminal-reservations")
    reservations.mkdir(parents=True)

    def uncertain(_path: Path) -> None:
        raise OSError("simulated directory fsync uncertainty")

    monkeypatch.setattr(research_git, "_fsync_directory", uncertain)
    with pytest.raises(OSError, match="directory fsync uncertainty"):
        _reserve(repository)

    recovered = _reserve(repository)
    assert recovered == reservations / f"{START_ID}.json"
    assert (
        _load_experiment_admission_terminal_reservation(
            repository.path, start_object_id=START_ID
        )
        is not None
    )
    with pytest.raises(ResearchGitError, match="conflicting attempt identity"):
        _reserve(repository, attempt_object_hash=OTHER_ATTEMPT_HASH)
