"""CLI for executable, signed, and recoverable self-evolution operations."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai_scientist.protocol.attestation import (
    AttestationError,
    sign_attestation,
    verify_attestation,
)
from ai_scientist.utils.evolution_artifacts import (
    EvolutionArtifactError,
    build_evolution_candidate_from_sources,
)
from ai_scientist.utils.evolution_deployment import (
    EvolutionDeploymentError,
    LocalEvolutionDeployment,
    deploy_approved_candidate,
)
from ai_scientist.utils.evolution_runtime import (
    EvolutionRuntimeError,
    run_canary_suite,
    run_shadow_benchmark,
)


def _load_json(path_value: str) -> Any:
    path = Path(path_value).expanduser()
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvolutionRuntimeError(f"JSON file could not be read: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvolutionRuntimeError(f"invalid JSON file {path}: {exc.msg}") from exc


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvolutionRuntimeError(f"{label} must be a JSON object")
    return value


def _candidate(value: Any) -> dict[str, Any]:
    payload = _object(value, label="candidate")
    nested = payload.get("candidate")
    return (
        _object(nested, label="candidate.candidate") if nested is not None else payload
    )


def _write_json(path_value: str | None, payload: Any, *, force: bool = False) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if not path_value:
        print(text, end="")
        return
    path = Path(path_value).expanduser().resolve()
    if path.exists() and not force:
        raise EvolutionRuntimeError("output path exists; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, raw_temp = tempfile.mkstemp(prefix=".evolution-", dir=str(path.parent))
    temp = Path(raw_temp)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _trust_store(path_value: str) -> dict[str, dict[str, Any]]:
    path = Path(path_value).expanduser().resolve()
    raw = _object(_load_json(str(path)), label="trust store")
    entries = raw.get("keys", raw)
    if not isinstance(entries, Mapping):
        raise AttestationError("trust store keys must be an object")
    result: dict[str, dict[str, Any]] = {}
    for key_id, raw_entry in entries.items():
        entry = _object(raw_entry, label=f"trust key {key_id}").copy()
        algorithm = entry.get("algorithm")
        if algorithm == "hmac-sha256":
            env_name = str(entry.pop("key_env", "") or "")
            if not env_name or not os.environ.get(env_name):
                raise AttestationError(
                    f"trusted HMAC key {key_id} requires populated key_env"
                )
            entry["key"] = os.environ[env_name].encode("utf-8")
        elif algorithm == "ed25519":
            public_key_file = str(entry.pop("public_key_file", "") or "")
            if not public_key_file:
                raise AttestationError(
                    f"trusted Ed25519 key {key_id} requires public_key_file"
                )
            key_path = (path.parent / public_key_file).resolve()
            try:
                entry["public_key"] = key_path.read_bytes()
            except OSError as exc:
                raise AttestationError(
                    f"trusted public key could not be read: {key_path}"
                ) from exc
        else:
            raise AttestationError(f"unsupported trusted algorithm for {key_id}")
        result[str(key_id)] = entry
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xscientist evolution",
        description=(
            "Build immutable candidates, execute controlled evidence, sign identities, "
            "and deploy only after verified authorization."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate = subparsers.add_parser(
        "candidate", help="Build baseline/candidate CAS artifacts from one JSON spec."
    )
    candidate.add_argument("--spec", required=True)
    candidate.add_argument("--constitution", required=True)
    candidate.add_argument("--store", required=True)
    candidate.add_argument("--out")
    candidate.add_argument("--force", action="store_true")

    benchmark = subparsers.add_parser(
        "benchmark", help="Run paired baseline/candidate evaluator commands."
    )
    benchmark.add_argument("--suite", required=True)
    benchmark.add_argument("--candidate", required=True)
    benchmark.add_argument("--store", required=True)
    benchmark.add_argument("--execute", action="store_true")
    benchmark.add_argument("--timeout", type=int, default=600)
    benchmark.add_argument("--out")
    benchmark.add_argument("--force", action="store_true")

    canary = subparsers.add_parser(
        "canary", help="Run bounded projects against a candidate and prove rollback."
    )
    canary.add_argument("--suite", required=True)
    canary.add_argument("--candidate", required=True)
    canary.add_argument("--store", required=True)
    canary.add_argument("--deployment-root", required=True)
    canary.add_argument("--executed-by", required=True)
    canary.add_argument("--execute", action="store_true")
    canary.add_argument("--timeout", type=int, default=600)
    canary.add_argument("--out")
    canary.add_argument("--force", action="store_true")

    attest = subparsers.add_parser("attest", help="Sign or verify one JSON artifact.")
    attest_subparsers = attest.add_subparsers(dest="attest_command", required=True)
    sign = attest_subparsers.add_parser("sign")
    sign.add_argument("--payload", required=True)
    sign.add_argument("--purpose", required=True)
    sign.add_argument("--identity", required=True)
    sign.add_argument("--key-id", required=True)
    signing_key = sign.add_mutually_exclusive_group(required=True)
    signing_key.add_argument("--key-env")
    signing_key.add_argument("--private-key")
    sign.add_argument("--out")
    sign.add_argument("--force", action="store_true")
    verify = attest_subparsers.add_parser("verify")
    verify.add_argument("--payload", required=True)
    verify.add_argument("--attestation", required=True)
    verify.add_argument("--trust-store", required=True)
    verify.add_argument("--purpose")
    verify.add_argument("--identity")

    deploy = subparsers.add_parser(
        "deploy", help="Plan or apply a signed, semantically approved candidate."
    )
    deploy.add_argument("--candidate", required=True)
    deploy.add_argument("--promotion", required=True)
    deploy.add_argument("--constitution", required=True)
    deploy.add_argument("--authorization", required=True)
    deploy.add_argument("--trust-store", required=True)
    deploy.add_argument("--store", required=True)
    deploy.add_argument("--deployment-root", required=True)
    deploy.add_argument("--target", required=True)
    deploy.add_argument("--executed-by", required=True)
    deploy.add_argument("--approval", required=True)
    deploy.add_argument("--apply", action="store_true")
    deploy.add_argument("--research-repo")
    deploy.add_argument("--promoted-object")
    deploy.add_argument("--no-commit", action="store_true")
    deploy.add_argument("--out")
    deploy.add_argument("--force", action="store_true")

    rollback = subparsers.add_parser(
        "rollback", help="Plan or restore the exact baseline artifact."
    )
    rollback.add_argument("--candidate", required=True)
    rollback.add_argument("--store", required=True)
    rollback.add_argument("--deployment-root", required=True)
    rollback.add_argument("--target", required=True)
    rollback.add_argument("--executed-by", required=True)
    rollback.add_argument("--approval", required=True)
    rollback.add_argument("--trigger", default="operator_rollback")
    rollback.add_argument("--production", action="store_true")
    rollback.add_argument("--authorization")
    rollback.add_argument("--trust-store")
    rollback.add_argument("--apply", action="store_true")
    rollback.add_argument("--research-repo")
    rollback.add_argument("--candidate-object")
    rollback.add_argument("--promoted-object")
    rollback.add_argument("--no-commit", action="store_true")
    rollback.add_argument("--out")
    rollback.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "candidate":
            spec = _object(_load_json(args.spec), label="candidate build spec")
            constitution = _object(
                _load_json(args.constitution), label="science constitution"
            )
            result = build_evolution_candidate_from_sources(
                constitution=constitution,
                store_root=args.store,
                **spec,
            )
            _write_json(args.out, result, force=args.force)
            return 0
        if args.command == "benchmark":
            result = run_shadow_benchmark(
                _object(_load_json(args.suite), label="benchmark suite"),
                _candidate(_load_json(args.candidate)),
                artifact_store=args.store,
                allow_execution=args.execute,
                timeout_seconds=args.timeout,
            )
            _write_json(args.out, result, force=args.force)
            return 0 if result["status"] == "completed" else 3
        if args.command == "canary":
            result = run_canary_suite(
                _object(_load_json(args.suite), label="canary suite"),
                _candidate(_load_json(args.candidate)),
                artifact_store=args.store,
                deployment_root=args.deployment_root,
                executed_by=args.executed_by,
                allow_execution=args.execute,
                timeout_seconds=args.timeout,
            )
            _write_json(args.out, result, force=args.force)
            return 0 if result["status"] == "passed" else 3
        if args.command == "attest" and args.attest_command == "sign":
            payload = _load_json(args.payload)
            if args.key_env:
                secret = os.environ.get(args.key_env)
                if not secret:
                    raise AttestationError("signing key environment variable is empty")
                algorithm = "hmac-sha256"
                key = secret.encode("utf-8")
            else:
                algorithm = "ed25519"
                try:
                    key = Path(args.private_key).expanduser().read_bytes()
                except OSError as exc:
                    raise AttestationError("private key could not be read") from exc
            result = sign_attestation(
                payload,
                purpose=args.purpose,
                identity=args.identity,
                key_id=args.key_id,
                algorithm=algorithm,
                key=key,
            )
            _write_json(args.out, result, force=args.force)
            return 0
        if args.command == "attest":
            result = verify_attestation(
                _object(_load_json(args.attestation), label="attestation"),
                _load_json(args.payload),
                trust_store=_trust_store(args.trust_store),
                purpose=args.purpose,
                identity=args.identity,
            )
            _write_json(None, result)
            return 0 if result["ok"] else 3
        if args.command == "deploy":
            candidate = _candidate(_load_json(args.candidate))
            deployment = LocalEvolutionDeployment(
                artifact_store=args.store,
                deployment_root=args.deployment_root,
                executed_by=args.executed_by,
            )
            result = deploy_approved_candidate(
                candidate,
                _object(_load_json(args.promotion), label="promotion"),
                constitution=_object(
                    _load_json(args.constitution), label="science constitution"
                ),
                authorization_bundle=_object(
                    _load_json(args.authorization), label="authorization bundle"
                ),
                trust_store=_trust_store(args.trust_store),
                deployment=deployment,
                target=args.target,
                approval_id=args.approval,
                apply=args.apply,
            )
            if bool(args.research_repo) is not bool(args.promoted_object):
                raise EvolutionDeploymentError(
                    "--research-repo and --promoted-object must be used together"
                )
            if args.research_repo:
                if not args.apply or not result.get("production_mutated"):
                    raise EvolutionDeploymentError(
                        "Research VCS recording requires an applied production deployment"
                    )
                from .research_evolution import ResearchEvolution

                recorded = ResearchEvolution(args.research_repo).deployment(
                    result,
                    promoted_id=args.promoted_object,
                    commit=not args.no_commit,
                )
                result["research_vcs"] = {
                    "decision": recorded["decision"].to_dict(),
                    "checkpoint": (
                        recorded["checkpoint"].to_dict()
                        if recorded["checkpoint"] is not None
                        else None
                    ),
                }
            _write_json(args.out, result, force=args.force)
            return 0
        if args.command == "rollback":
            candidate = _candidate(_load_json(args.candidate))
            authorization = None
            if args.production:
                if not args.authorization or not args.trust_store:
                    raise EvolutionDeploymentError(
                        "production rollback requires --authorization and --trust-store"
                    )
                authorization_payload = {
                    "candidate_hash": candidate.get("candidate_hash"),
                    "baseline_artifact_hash": candidate.get("base_artifact_hash"),
                    "target": args.target,
                    "trigger": args.trigger,
                }
                authorization = verify_attestation(
                    _object(
                        _load_json(args.authorization),
                        label="rollback authorization",
                    ),
                    authorization_payload,
                    trust_store=_trust_store(args.trust_store),
                    purpose="production_rollback",
                    identity=args.approval,
                )
                if not authorization["ok"]:
                    raise EvolutionDeploymentError(
                        "rollback authorization failed: "
                        + ", ".join(authorization["errors"])
                    )
            deployment = LocalEvolutionDeployment(
                artifact_store=args.store,
                deployment_root=args.deployment_root,
                executed_by=args.executed_by,
            )
            result = deployment.rollback(
                candidate,
                target=args.target,
                apply=args.apply,
                approval_id=args.approval,
                production=args.production,
                trigger=args.trigger,
                authorization=authorization,
            )
            record_values = (
                args.research_repo,
                args.candidate_object,
                args.promoted_object,
            )
            if any(record_values) and not all(record_values):
                raise EvolutionDeploymentError(
                    "rollback recording requires --research-repo, --candidate-object, "
                    "and --promoted-object"
                )
            if args.research_repo:
                if not args.apply or not args.production:
                    raise EvolutionDeploymentError(
                        "Research VCS rollback recording requires --production --apply"
                    )
                from .research_evolution import ResearchEvolution

                recorded = ResearchEvolution(args.research_repo).rollback(
                    result["rollback_receipt"],
                    candidate_id=args.candidate_object,
                    promoted_id=args.promoted_object,
                    trigger=args.trigger,
                    commit=not args.no_commit,
                )
                result["research_vcs"] = {
                    "decision": recorded["decision"].to_dict(),
                    "checkpoint": (
                        recorded["checkpoint"].to_dict()
                        if recorded["checkpoint"] is not None
                        else None
                    ),
                }
            _write_json(args.out, result, force=args.force)
            return 0
    except (
        AttestationError,
        EvolutionArtifactError,
        EvolutionDeploymentError,
        EvolutionRuntimeError,
        OSError,
        TypeError,
        ValueError,
    ) as exc:
        payload = {
            "schema_version": "xscientist.error.v1",
            "ok": False,
            "error": {
                "category": exc.__class__.__name__,
                "command": args.command,
                "message": str(exc),
            },
        }
        print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)
        return 2
    parser.error(f"unsupported evolution command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
