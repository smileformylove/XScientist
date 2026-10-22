from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

from ai_scientist.protocol.attestation import sign_attestation
from xscientist import ResearchLifecycle, ResearchRepository
from xscientist import experiment_admission_runtime as runtime
from xscientist.experiment_admission import (
    EXPERIMENT_ADMISSION_PROTOCOL,
    XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
)
from xscientist.research_git import (
    ResearchGitError,
    _repository_common_git_dir,
    capture_environment_receipt,
    create_checkpoint,
)

_SECRET = "sigkill-secret-api-key-must-not-survive"
_PHASES = (
    "after_claim",
    "before_bfts_terminal",
    "after_bfts_terminal",
    "before_worker_terminal",
    "after_worker_terminal",
)

_CHILD_PROGRAM = r"""
import json
import os
import signal
import sys
from pathlib import Path

from xscientist import experiment_admission_runtime as runtime

repo = Path(sys.argv[1])
receipt = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
experiment_dir = Path(sys.argv[3])
phase = sys.argv[4]
ready_path = Path(sys.argv[5])
secret = os.environ["XSCIENTIST_SIGKILL_TEST_SECRET"]


def stop_at(label):
    if phase != label:
        return
    descriptor = os.open(ready_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (label + "\n").encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory = os.open(ready_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    while True:
        signal.pause()


claim = runtime.claim_experiment_admission_start_v2(
    repo,
    receipt,
    experiment_dir=experiment_dir,
    idea_key="sigkill-idea",
    run_nonce="1" * 32,
    worker_nonce="2" * 32,
    bfts_run_nonce="3" * 32,
    owner_nonce="4" * 32,
)
stop_at("after_claim")

checkpoint = experiment_dir / "checkpoint.json"
checkpoint.write_text('{"step":1}\n', encoding="utf-8")
with checkpoint.open("rb") as stream:
    os.fsync(stream.fileno())
stop_at("before_bfts_terminal")

runtime.publish_bfts_terminal(
    repo,
    claim,
    status="completed",
    stage="bfts",
    resumable=False,
    failure={
        "type": "SensitiveBFTSError",
        "message": secret,
        "traceback": secret,
    },
    artifact_paths=[checkpoint],
)
stop_at("after_bfts_terminal")

manager_state = experiment_dir / "manager_state.json"
manager_state.write_text('{"state":"complete"}\n', encoding="utf-8")
with manager_state.open("rb") as stream:
    os.fsync(stream.fileno())
stop_at("before_worker_terminal")

runtime.publish_worker_terminal(
    repo,
    claim,
    {
        "idea_idx": claim["idea_index"],
        "status": "success",
        "registry_status": "completed",
        "research_status": "recording",
        "stage": "evidence",
        "exp_dir": str(experiment_dir),
        "checkpoint_path": str(checkpoint),
        "manager_state_path": str(manager_state),
        "error": secret,
        "traceback": secret,
    },
    provenance={"environment_hash": "sha256:" + "a" * 64},
    failure={
        "type": "SensitiveWorkerError",
        "message": secret,
        "traceback": secret,
    },
)
stop_at("after_worker_terminal")
raise RuntimeError("unreachable phase")
"""


def _signed_admission_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ResearchRepository, dict[str, Any], Path]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError:
        pytest.skip("cryptography is required for a real admission signature")

    repository = ResearchRepository.init(
        tmp_path / "research",
        question="Can runtime truth survive an uncatchable process death?",
        git_user_name="SIGKILL Runtime Test",
        git_user_email="sigkill-runtime@example.invalid",
    )
    config_path = repository.path / "research.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["experiment_admission"] = {"mode": "required"}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    hypothesis = repository.record(
        "hypothesis",
        {"statement": "Durable outboxes preserve terminal experiment truth."},
    )
    plan = repository.record(
        "research_plan",
        {
            "summary": "Exercise only marker and artifact durability.",
            "hypothesis_id": hypothesis.object_id,
        },
        state="locked",
        relations=[
            {
                "type": "depends_on",
                "target": hypothesis.object_id,
                "role": "hypothesis",
            }
        ],
    )
    repository.commit(stage="plan", subject="lock SIGKILL durability fixture")

    scope = {
        "idea_index": 0,
        "ideas_hash": "sha256:" + "5" * 64,
        "configuration_hash": "sha256:" + "6" * 64,
        "budget_hash": "sha256:" + "7" * 64,
        "seeds": [7],
        "environment_hash": capture_environment_receipt(repository.path)[
            "content_hash"
        ],
    }
    lifecycle = ResearchLifecycle(repository)
    admission_payload = lifecycle.prepare_experiment_admission_payload(
        plan_id=plan.object_id,
        execution_scope=scope,
        decision="authorize",
    )

    authority = tmp_path / "external-authority"
    authority.mkdir()
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_key = authority / "reviewer.pub"
    public_key.write_bytes(public_pem)
    identity = "human:sigkill-independent-reviewer"
    key_id = "key:sigkill-independent-reviewer"
    trust_store = authority / "trust.json"
    trust_store.write_text(
        json.dumps(
            {
                "keys": {
                    key_id: {
                        "identity": identity,
                        "algorithm": "ed25519",
                        "public_key_file": public_key.name,
                        "revoked": False,
                    }
                }
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        XSCIENTIST_EXPERIMENT_ADMISSION_TRUST_STORE,
        str(trust_store),
    )
    attestation = sign_attestation(
        admission_payload,
        purpose=EXPERIMENT_ADMISSION_PROTOCOL,
        identity=identity,
        key_id=key_id,
        algorithm="ed25519",
        key=private_pem,
        issued_at="2026-09-05T00:00:00+00:00",
    )
    lifecycle.ingest_experiment_admission(
        admission_payload,
        attestation=attestation,
        trust_store=trust_store,
    )

    issued = lifecycle.issue_experiment_admission_start(
        plan_id=plan.object_id,
        execution_scope=scope,
    )
    start_path = issued["start"].path.relative_to(repository.path).as_posix()
    checkpoint = create_checkpoint(
        repository.path,
        stage="experiment_admission_start",
        subject="commit SIGKILL admission start",
        status="locked",
        only_paths=[start_path],
    )
    assert checkpoint.commit
    receipt = lifecycle.finalize_experiment_admission_start(
        issued["receipt"],
        start_checkpoint=checkpoint.commit,
    )
    experiment_dir = repository.path / "02_experiments" / "sigkill-run"
    experiment_dir.mkdir(parents=True)
    return repository, receipt, experiment_dir


def _wait_until_ready(process: subprocess.Popen[str], ready_path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if ready_path.is_file():
            return
        return_code = process.poll()
        if return_code is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"child exited before kill point ({return_code}): {stdout!r} {stderr!r}"
            )
        time.sleep(0.01)
    raise AssertionError("child did not reach the SIGKILL point")


def _state_after_sigkill_before_reap(claim: dict[str, Any]) -> str:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        state = runtime.execution_claim_owner_state(claim)
        if state != "alive":
            return state
        time.sleep(0.01)
    return "alive"


def _marker_path(repository: ResearchRepository, directory: str, start_id: str) -> Path:
    return (
        _repository_common_git_dir(repository.path)
        / "xscientist"
        / directory
        / f"{start_id}.json"
    )


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "SIGKILL"),
    reason="real SIGKILL durability requires POSIX",
)
@pytest.mark.parametrize("phase", _PHASES)
def test_runtime_markers_survive_real_sigkill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    repository, receipt, experiment_dir = _signed_admission_start(
        tmp_path,
        monkeypatch,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    ready_path = tmp_path / "child-ready"
    child_env = os.environ.copy()
    child_env["XSCIENTIST_SIGKILL_TEST_SECRET"] = _SECRET
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD_PROGRAM,
            str(repository.path),
            str(receipt_path),
            str(experiment_dir),
            phase,
            str(ready_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_until_ready(process, ready_path)
        claim = runtime.load_execution_claim_v2(
            repository.path,
            start_object_id=str(receipt["start_object_id"]),
        )
        assert claim is not None
        assert runtime.execution_claim_owner_state(claim) == "alive"

        os.kill(process.pid, signal.SIGKILL)
        pre_reap_state = _state_after_sigkill_before_reap(claim)
        assert pre_reap_state in {"dead", "unknown"}
        assert process.wait(timeout=10) == -signal.SIGKILL
        stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            os.kill(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    assert _SECRET not in stdout
    assert _SECRET not in stderr
    records = runtime.admission_runtime_records(repository.path)
    assert len(records) == 1
    record = records[0]
    assert record["owner_state"] == "dead"
    assert runtime.execution_claim_owner_state(record["claim"]) == "dead"

    before_bfts = phase in {"after_claim", "before_bfts_terminal"}
    after_worker = phase == "after_worker_terminal"
    if before_bfts:
        assert record["bfts_terminal"] is None
        assert record["worker_terminal"] is None
    elif after_worker:
        assert record["bfts_terminal"] is not None
        assert record["worker_terminal"] is not None
    else:
        assert record["bfts_terminal"] is not None
        assert record["worker_terminal"] is None

    checkpoint_path = experiment_dir / "checkpoint.json"
    manager_state_path = experiment_dir / "manager_state.json"
    assert checkpoint_path.is_file() is (phase != "after_claim")
    assert manager_state_path.is_file() is (
        phase in {"before_worker_terminal", "after_worker_terminal"}
    )
    if checkpoint_path.is_file():
        assert checkpoint_path.read_text(encoding="utf-8") == '{"step":1}\n'
    if manager_state_path.is_file():
        assert manager_state_path.read_text(encoding="utf-8") == (
            '{"state":"complete"}\n'
        )

    recovered = runtime.result_from_runtime_record(repository.path, record)
    assert recovered is not None
    result, provenance = recovered
    if before_bfts:
        assert result["runtime_status"] == "interrupted"
        assert result["registry_status"] == "failed"
        assert result["stage"] == "worker_lost"
        assert provenance == {}
    elif after_worker:
        assert result["status"] == "success"
        assert result["registry_status"] == "completed"
        assert result["stage"] == "evidence"
        assert provenance == {"environment_hash": "sha256:" + "a" * 64}
    else:
        assert result["status"] == "failed"
        assert result["registry_status"] == "completed"
        assert result["stage"] == "worker_lost_after_bfts"
        assert provenance == {}

    state_root = _repository_common_git_dir(repository.path) / "xscientist"
    persisted = b"\n".join(
        path.read_bytes() for path in sorted(state_root.rglob("*.json"))
    )
    assert _SECRET.encode("utf-8") not in persisted
    assert _SECRET not in json.dumps(record, sort_keys=True)
    assert _SECRET not in json.dumps(result, sort_keys=True)

    claim_path = _marker_path(
        repository,
        "admission-claims",
        str(receipt["start_object_id"]),
    )
    claim_bytes = claim_path.read_bytes()
    with pytest.raises(ResearchGitError, match="already been claimed"):
        runtime.claim_experiment_admission_start_v2(
            repository.path,
            receipt,
            experiment_dir=experiment_dir,
            idea_key="sigkill-idea",
        )
    assert claim_path.read_bytes() == claim_bytes

    if record["bfts_terminal"] is not None:
        bfts_path = _marker_path(
            repository,
            "admission-bfts-terminals",
            str(receipt["start_object_id"]),
        )
        bfts_bytes = bfts_path.read_bytes()
        with pytest.raises(ResearchGitError, match="conflicts"):
            runtime.publish_bfts_terminal(
                repository.path,
                claim,
                status="failed",
                stage="conflicting_retry",
                resumable=False,
            )
        assert bfts_path.read_bytes() == bfts_bytes

    if record["worker_terminal"] is not None:
        worker_path = _marker_path(
            repository,
            "admission-worker-terminals",
            str(receipt["start_object_id"]),
        )
        worker_bytes = worker_path.read_bytes()
        with pytest.raises(ResearchGitError, match="conflicts"):
            runtime.publish_worker_terminal(
                repository.path,
                claim,
                {
                    "idea_idx": claim["idea_index"],
                    "status": "failed",
                    "registry_status": "failed",
                    "stage": "conflicting_retry",
                },
            )
        assert worker_path.read_bytes() == worker_bytes
