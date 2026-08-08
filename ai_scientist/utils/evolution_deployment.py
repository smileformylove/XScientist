"""Recoverable local deployment adapter for approved evolution artifacts."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ai_scientist.protocol.attestation import verify_authorization_bundle
from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.utils.evolution_artifacts import (
    EvolutionArtifactError,
    materialize_evolution_artifact,
    verify_evolution_artifact,
)
from ai_scientist.utils.evolution_gate import (
    build_rollback_receipt,
    validate_evolution_candidate,
    validate_production_promotion,
)

DEPLOYMENT_RECEIPT_SCHEMA = "xscientist.evolution-deployment.v1"
_STATE_DIR = ".xscientist-deploy"
_MARKER = ".xscientist-deployment.json"


class EvolutionDeploymentError(RuntimeError):
    """Raised when a deployment would be unsafe or cannot be verified."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_target(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip().strip("/")
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", "..", _STATE_DIR} for part in path.parts)
    ):
        raise EvolutionDeploymentError("deployment target must be a safe relative path")
    return path.as_posix()


def _hash_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    if not text.startswith("sha256:") or len(text) != 71:
        return False
    digest = text[7:]
    return digest == digest.lower() and all(
        char in "0123456789abcdef" for char in digest
    )


def _tree_hash(root: Path | None) -> str | None:
    if root is None or not root.exists():
        return None
    if root.is_symlink() or not root.is_dir():
        raise EvolutionDeploymentError(
            "managed deployment target is not a real directory"
        )
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise EvolutionDeploymentError("managed target contains a symbolic link")
        if path.is_dir():
            continue
        if path.name == _MARKER:
            continue
        if not path.is_file():
            raise EvolutionDeploymentError("managed target contains a special file")
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _hash_file(path),
                "executable": bool(path.stat().st_mode & 0o111),
            }
        )
    return canonical_content_hash(entries)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=".receipt-", dir=str(path.parent))
    temp = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def _deployment_lock(state_root: Path, *, timeout_seconds: float = 30.0):
    """Serialize cooperating deployers and recover abandoned lock files."""

    lock_path = state_root / "deployment.lock"
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            try:
                stale = time.time() - lock_path.stat().st_mtime > 600
            except OSError:
                stale = False
            if stale:
                stale_path = state_root / f"deployment.lock.stale-{int(time.time())}"
                try:
                    os.replace(lock_path, stale_path)
                except OSError:
                    pass
                continue
            if time.monotonic() >= deadline:
                raise EvolutionDeploymentError("timed out waiting for deployment lock")
            time.sleep(0.05)
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()}\ncreated_at={_now_iso()}\n")
                handle.flush()
                os.fsync(handle.fileno())
            break
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _verify_materialized(
    target: Path, artifact_store: str | Path, artifact_hash: str
) -> dict[str, Any]:
    check = verify_evolution_artifact(artifact_store, artifact_hash)
    if not check["ok"]:
        return {"ok": False, "errors": check["errors"]}
    manifest = check["manifest"]
    prefix = str(manifest["logical_root"]).rstrip("/") + "/"
    expected: dict[str, dict[str, Any]] = {}
    for entry in manifest["entries"]:
        path = str(entry["path"])
        if not path.startswith(prefix):
            return {"ok": False, "errors": ["entry_outside_logical_root"]}
        expected[path[len(prefix) :]] = entry
    errors: list[str] = []
    actual_paths: set[str] = set()
    for path in sorted(target.rglob("*")):
        if path.is_symlink():
            errors.append("symbolic_link_present")
            continue
        if path.is_dir() or path.name == _MARKER:
            continue
        relative = path.relative_to(target).as_posix()
        actual_paths.add(relative)
        entry = expected.get(relative)
        if entry is None:
            errors.append(f"unexpected_file:{relative}")
            continue
        if _hash_file(path) != entry["sha256"]:
            errors.append(f"file_hash_mismatch:{relative}")
        if bool(path.stat().st_mode & 0o111) is not entry["executable"]:
            errors.append(f"file_mode_mismatch:{relative}")
    for missing in sorted(set(expected) - actual_paths):
        errors.append(f"file_missing:{missing}")
    return {"ok": not errors, "errors": errors, "tree_hash": _tree_hash(target)}


def validate_deployment_receipt(
    payload: Mapping[str, Any], *, candidate_artifact_hash: str | None = None
) -> dict[str, Any]:
    """Validate an applied receipt before it enters Research VCS history."""

    receipt = {
        key: value
        for key, value in dict(payload or {}).items()
        if key != "receipt_path"
    }
    errors: list[str] = []
    expected_fields = {
        "schema_version",
        "mode",
        "generated_at",
        "target",
        "candidate_artifact_hash",
        "before_tree_hash",
        "after_tree_hash",
        "backup_ref",
        "executed_by",
        "approval_id",
        "authorization_hash",
        "status",
        "production_mutated",
        "receipt_hash",
    }
    if set(receipt) != expected_fields:
        errors.append("deployment_receipt_fields_invalid")
    if receipt.get("schema_version") != DEPLOYMENT_RECEIPT_SCHEMA:
        errors.append("schema_version_invalid")
    if receipt.get("mode") not in {"canary", "production"}:
        errors.append("deployment_mode_invalid")
    if receipt.get("status") != "applied":
        errors.append("deployment_not_applied")
    if receipt.get("production_mutated") is not (receipt.get("mode") == "production"):
        errors.append("production_mutation_flag_invalid")
    try:
        parsed_time = datetime.fromisoformat(
            str(receipt.get("generated_at") or "").replace("Z", "+00:00")
        )
        if parsed_time.tzinfo is None:
            raise ValueError("timezone missing")
    except ValueError:
        errors.append("generated_at_invalid")
    try:
        _safe_target(str(receipt.get("target") or ""))
    except EvolutionDeploymentError:
        errors.append("deployment_target_invalid")
    backup_ref = receipt.get("backup_ref")
    if backup_ref is not None:
        backup_path = PurePosixPath(str(backup_ref).replace("\\", "/"))
        if (
            backup_path.is_absolute()
            or any(part in {"", ".", ".."} for part in backup_path.parts)
            or backup_path.parts[:2] != (_STATE_DIR, "backups")
        ):
            errors.append("backup_ref_invalid")
    if not str(receipt.get("executed_by") or "").startswith(("service:", "human:")):
        errors.append("deployment_executor_invalid")
    if not str(receipt.get("approval_id") or "").startswith("human:"):
        errors.append("deployment_approval_invalid")
    for field in ("candidate_artifact_hash", "after_tree_hash"):
        if not _is_sha256(receipt.get(field)):
            errors.append(f"{field}_invalid")
    if receipt.get("before_tree_hash") is not None and not _is_sha256(
        receipt.get("before_tree_hash")
    ):
        errors.append("before_tree_hash_invalid")
    if (
        candidate_artifact_hash is not None
        and receipt.get("candidate_artifact_hash") != candidate_artifact_hash
    ):
        errors.append("candidate_artifact_binding_mismatch")
    if receipt.get("mode") == "production" and not _is_sha256(
        receipt.get("authorization_hash")
    ):
        errors.append("production_authorization_missing")
    receipt_hash = receipt.pop("receipt_hash", None)
    try:
        if receipt_hash != canonical_content_hash(receipt):
            errors.append("deployment_receipt_hash_mismatch")
    except (TypeError, ValueError):
        errors.append("deployment_receipt_not_canonicalizable")
    return {
        "ok": not errors,
        "errors": errors,
        "receipt_hash": receipt_hash,
        "receipt": {**receipt, "receipt_hash": receipt_hash},
    }


class LocalEvolutionDeployment:
    """Atomic directory-swap adapter confined to one explicit deployment root."""

    def __init__(
        self,
        *,
        artifact_store: str | Path,
        deployment_root: str | Path,
        executed_by: str,
    ) -> None:
        self.artifact_store = Path(artifact_store).expanduser().resolve()
        self.deployment_root = Path(deployment_root).expanduser().resolve()
        self.executed_by = str(executed_by or "").strip()
        if not self.executed_by.startswith(("service:", "human:")):
            raise EvolutionDeploymentError(
                "deployment executor must use service: or human: identity"
            )
        home = Path.home().resolve()
        if self.deployment_root in {Path(self.deployment_root.anchor), home}:
            raise EvolutionDeploymentError(
                "filesystem root and home directory cannot be deployment roots"
            )
        self.deployment_root.mkdir(parents=True, exist_ok=True)
        if self.deployment_root.is_symlink():
            raise EvolutionDeploymentError("deployment root may not be a symbolic link")
        self.state_root = self.deployment_root / _STATE_DIR
        self.state_root.mkdir(exist_ok=True)

    def _target(self, target: str) -> tuple[str, Path]:
        relative = _safe_target(target)
        resolved = (self.deployment_root / Path(relative)).resolve()
        if self.deployment_root not in resolved.parents:
            raise EvolutionDeploymentError("deployment target escapes its root")
        return relative, resolved

    def plan(self, *, target: str, artifact_hash: str) -> dict[str, Any]:
        relative, target_path = self._target(target)
        check = verify_evolution_artifact(self.artifact_store, artifact_hash)
        if not check["ok"]:
            raise EvolutionDeploymentError(
                "candidate artifact invalid: " + ", ".join(check["errors"])
            )
        marker: dict[str, Any] = {}
        marker_path = target_path / _MARKER
        if marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                marker = {"invalid": True}
        return {
            "schema_version": DEPLOYMENT_RECEIPT_SCHEMA,
            "mode": "plan",
            "target": relative,
            "target_exists": target_path.exists(),
            "current_artifact_hash": marker.get("artifact_hash"),
            "current_tree_hash": _tree_hash(target_path),
            "candidate_artifact_hash": artifact_hash,
            "candidate_file_count": check["manifest"]["file_count"],
            "candidate_total_bytes": check["manifest"]["total_bytes"],
            "will_replace": marker.get("artifact_hash") != artifact_hash,
            "production_mutated": False,
        }

    def deploy(
        self,
        *,
        target: str,
        artifact_hash: str,
        apply: bool = False,
        approval_id: str | None = None,
        authorization: Mapping[str, Any] | None = None,
        production: bool = False,
    ) -> dict[str, Any]:
        plan = self.plan(target=target, artifact_hash=artifact_hash)
        if not apply:
            return plan
        if not str(approval_id or "").startswith("human:"):
            raise EvolutionDeploymentError("apply requires an explicit human: approval")
        if production and not (authorization or {}).get("ok"):
            raise EvolutionDeploymentError(
                "production apply requires a verified signed authorization bundle"
            )
        with _deployment_lock(self.state_root):
            plan = self.plan(target=target, artifact_hash=artifact_hash)
            relative, target_path = self._target(target)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.is_symlink():
                raise EvolutionDeploymentError(
                    "deployment target may not be a symbolic link"
                )
            staging_parent = self.state_root / "staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix="release-", dir=str(staging_parent)))
            shutil.rmtree(staging)
            materialize_evolution_artifact(
                self.artifact_store,
                artifact_hash,
                staging,
                strip_logical_root=True,
            )
            marker = {
                "schema_version": DEPLOYMENT_RECEIPT_SCHEMA,
                "artifact_hash": artifact_hash,
                "deployed_at": _now_iso(),
                "executed_by": self.executed_by,
                "approval_id": approval_id,
                "production": bool(production),
            }
            _atomic_json(staging / _MARKER, marker)
            backup: Path | None = None
            if target_path.exists():
                backup_parent = self.state_root / "backups" / Path(relative)
                backup_parent.mkdir(parents=True, exist_ok=True)
                backup = backup_parent / marker["deployed_at"].replace(":", "-")
                os.replace(target_path, backup)
            try:
                os.replace(staging, target_path)
                verification = _verify_materialized(
                    target_path, self.artifact_store, artifact_hash
                )
                if not verification["ok"]:
                    raise EvolutionDeploymentError(
                        "deployed artifact verification failed: "
                        + ", ".join(verification["errors"])
                    )
            except Exception:
                failed_parent = self.state_root / "failed"
                failed_parent.mkdir(parents=True, exist_ok=True)
                if target_path.exists():
                    failed = Path(
                        tempfile.mkdtemp(prefix="failed-", dir=str(failed_parent))
                    )
                    shutil.rmtree(failed)
                    os.replace(target_path, failed)
                if backup is not None and backup.exists():
                    os.replace(backup, target_path)
                raise
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        receipt = {
            "schema_version": DEPLOYMENT_RECEIPT_SCHEMA,
            "mode": "production" if production else "canary",
            "generated_at": _now_iso(),
            "target": relative,
            "candidate_artifact_hash": artifact_hash,
            "before_tree_hash": plan["current_tree_hash"],
            "after_tree_hash": verification["tree_hash"],
            "backup_ref": (
                backup.relative_to(self.deployment_root).as_posix()
                if backup is not None
                else None
            ),
            "executed_by": self.executed_by,
            "approval_id": approval_id,
            "authorization_hash": (
                canonical_content_hash(authorization) if authorization else None
            ),
            "status": "applied",
            "production_mutated": bool(production),
        }
        receipt["receipt_hash"] = canonical_content_hash(receipt)
        receipt_path = (
            self.state_root
            / "receipts"
            / (receipt["receipt_hash"].split(":", 1)[1] + ".json")
        )
        _atomic_json(receipt_path, receipt)
        return {
            **receipt,
            "receipt_path": receipt_path.relative_to(self.deployment_root).as_posix(),
        }

    def rollback(
        self,
        candidate: Mapping[str, Any],
        *,
        target: str,
        apply: bool = False,
        approval_id: str | None = None,
        production: bool = False,
        trigger: str = "canary_exercise",
        authorization: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        check = validate_evolution_candidate(dict(candidate))
        if not check["ok"]:
            raise EvolutionDeploymentError(
                "candidate invalid: " + ", ".join(check["errors"])
            )
        if production and not (authorization or {}).get("ok"):
            raise EvolutionDeploymentError(
                "production rollback requires a verified signed authorization"
            )
        if production and authorization.get("identity") != approval_id:
            raise EvolutionDeploymentError(
                "production rollback signer must match the human approval identity"
            )
        deployment = self.deploy(
            target=target,
            artifact_hash=str(candidate["base_artifact_hash"]),
            apply=apply,
            approval_id=approval_id,
            authorization=authorization if production else None,
            production=production,
        )
        if not apply:
            return deployment
        execution_log = {
            key: value
            for key, value in deployment.items()
            if key not in {"receipt_path"}
        }
        rollback_receipt = build_rollback_receipt(
            dict(candidate),
            restored_artifact_hash=str(candidate["base_artifact_hash"]),
            execution_log_hash=canonical_content_hash(execution_log),
            executed_by=self.executed_by,
            trigger=trigger,
            exercise_only=not production,
        )
        return {
            "deployment": deployment,
            "rollback_receipt": rollback_receipt,
            "production_mutated": bool(production),
        }


def deploy_approved_candidate(
    candidate: Mapping[str, Any],
    promotion: Mapping[str, Any],
    *,
    constitution: Mapping[str, Any],
    authorization_bundle: Mapping[str, Any],
    trust_store: Mapping[str, Mapping[str, Any]],
    deployment: LocalEvolutionDeployment,
    target: str,
    approval_id: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate semantic promotion and signed authorities before file mutation."""

    promotion_check = validate_production_promotion(
        dict(promotion), constitution=dict(constitution)
    )
    if not promotion_check["passed"]:
        raise EvolutionDeploymentError(
            "promotion semantics invalid: " + ", ".join(promotion_check["errors"])
        )
    required_humans = 2 if candidate.get("risk_tier") == "high" else 1
    authorization = verify_authorization_bundle(
        authorization_bundle,
        candidate=candidate,
        promotion=promotion,
        trust_store=trust_store,
        minimum_human_approvers=required_humans,
    )
    if not authorization["ok"]:
        raise EvolutionDeploymentError(
            "signed deployment authorization failed: "
            + ", ".join(authorization["errors"])
        )
    if approval_id not in authorization["human_approvers"]:
        raise EvolutionDeploymentError(
            "deployment approval identity lacks a verified production signature"
        )
    authorization = {
        **authorization,
        "authorization_bundle_hash": canonical_content_hash(authorization_bundle),
    }
    return deployment.deploy(
        target=target,
        artifact_hash=str(candidate["candidate_artifact_hash"]),
        apply=apply,
        approval_id=approval_id,
        authorization=authorization,
        production=True,
    )


__all__ = [
    "DEPLOYMENT_RECEIPT_SCHEMA",
    "EvolutionDeploymentError",
    "LocalEvolutionDeployment",
    "deploy_approved_candidate",
    "validate_deployment_receipt",
]
