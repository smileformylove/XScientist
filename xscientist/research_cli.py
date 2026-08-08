"""Command-line interface for native, local-first research version control."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ai_scientist.protocol.research_vcs import (
    RESEARCH_OBJECT_KINDS,
    RESEARCH_OBJECT_STATES,
    RESEARCH_RELATION_TYPES,
)
from ai_scientist.utils.privacy import (
    portable_path,
    redact_sensitive_payload,
    redact_sensitive_text,
)

from .research_git import (
    ResearchGitError,
    add_research_object,
    commit_research_stage,
    create_checkpoint,
    create_research_branch,
    create_research_bundle,
    create_research_tag,
    delete_research_branch,
    init_repository,
    list_research_branches,
    list_research_objects,
    list_research_tags,
    load_research_object,
    merge_research_branch,
    preview_research_merge,
    record_research_object,
    rename_research_branch,
    research_blame,
    repository_status,
    reproduce_checkpoint,
    restore_research_bundle,
    restore_research_paths,
    revert_research_checkpoint,
    research_diff,
    research_log,
    research_stage,
    research_unstage,
    show_checkpoint,
    switch_research_branch,
    verify_research_bundle,
    verify_research_repository,
)


def _print_json(payload: Any) -> None:
    print(
        json.dumps(
            redact_sensitive_payload(payload),
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )


def _display_path(value: Any) -> str:
    return portable_path(str(value), base=Path.cwd())


def _display_text(value: Any) -> str:
    return redact_sensitive_text(str(value))


def _read_question(args: argparse.Namespace) -> str | None:
    if args.question and args.question_file:
        raise ResearchGitError("use only one of --question or --question-file")
    if args.question_file:
        path = Path(args.question_file).expanduser()
        if not path.is_file():
            raise ResearchGitError(f"question file not found: {path}")
        return path.read_text(encoding="utf-8")
    if args.question:
        return f"# Research question\n\n{args.question.strip()}\n"
    return None


def _read_object_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.data is not None and args.file is not None:
        raise ResearchGitError("use only one of --data or --file")
    if args.file is not None:
        path = Path(args.file).expanduser()
        if not path.is_file():
            raise ResearchGitError("research object payload file was not found")
        raw = path.read_text(encoding="utf-8")
    elif args.data is not None:
        raw = args.data
    else:
        raise ResearchGitError("record requires --data or --file")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResearchGitError(
            f"research object payload is invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise ResearchGitError("research object payload must be a JSON object")
    return payload


def _parse_relations(values: Sequence[str]) -> list[dict[str, str]]:
    relations: list[dict[str, str]] = []
    for value in values:
        relation_type, separator, remainder = value.partition(":")
        target, role_separator, role = remainder.partition(":")
        if not separator or relation_type not in RESEARCH_RELATION_TYPES or not target:
            raise ResearchGitError(
                "relation must be TYPE:TARGET[:ROLE] using a supported relation type"
            )
        relation = {"type": relation_type, "target": target}
        if role_separator and role:
            relation["role"] = role
        relations.append(relation)
    return relations


def _parse_assignments(values: Sequence[str], *, label: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for value in values:
        name, separator, raw = value.partition("=")
        name = name.strip()
        if not separator or not name or not raw.strip():
            raise ResearchGitError(f"{label} must use NAME=VALUE")
        if name in parsed:
            raise ResearchGitError(f"duplicate {label} name: {name}")
        try:
            parsed[name] = json.loads(raw)
        except json.JSONDecodeError:
            parsed[name] = raw.strip()
    return parsed


def _hash_local_file(path_value: str) -> str:
    import hashlib

    path = Path(path_value).expanduser()
    if not path.is_file():
        raise ResearchGitError("dataset split file was not found")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ResearchGitError("dataset split file could not be read") from exc
    return "sha256:" + digest.hexdigest()


def _print_saved_object(label: str, result: dict[str, Any], *, as_json: bool) -> None:
    recorded = result["object"]
    related = result.get("related") or []
    checkpoint = result.get("checkpoint")
    payload = {
        "object": recorded.to_dict(),
        "related_objects": [item.to_dict() for item in related],
        "checkpoint": checkpoint.to_dict() if checkpoint is not None else None,
    }
    if as_json:
        _print_json(payload)
        return
    action = "Recorded" if recorded.created else "Reused"
    print(f"{action} {label}: {recorded.object_id} ({recorded.state})")
    for item in related:
        related_action = "Recorded" if item.created else "Reused"
        print(
            f"{related_action} related {item.kind}: " f"{item.object_id} ({item.state})"
        )
    if checkpoint is None:
        print("Checkpoint: not requested")
    elif checkpoint.committed:
        print(f"Checkpoint: {checkpoint.checkpoint_id}")
    else:
        print(f"Checkpoint skipped: {checkpoint.reason}")


def _build_parser(*, prog: str = "xscientist research") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Version scientific questions, hypotheses, evidence, evaluations, and "
            "manuscripts locally. No server is required and no command pushes automatically."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check the local version-control backend and required capabilities.",
    )
    doctor_parser.add_argument("--json", action="store_true", dest="as_json")

    hypothesis_parser = subparsers.add_parser(
        "hypothesis",
        help="Record and checkpoint one falsifiable hypothesis.",
    )
    hypothesis_parser.add_argument("statement")
    hypothesis_parser.add_argument("--falsifier", required=True)
    hypothesis_parser.add_argument("--rationale", default="")
    hypothesis_parser.add_argument("--prediction", action="append", default=[])
    hypothesis_parser.add_argument("--repo", default=".")
    hypothesis_parser.add_argument("-m", "--message")
    hypothesis_parser.add_argument("--no-commit", action="store_true")
    hypothesis_parser.add_argument("--json", action="store_true", dest="as_json")

    preregistration_parser = subparsers.add_parser(
        "preregister",
        help="Lock a confirmatory plan and dataset split before experiments run.",
    )
    preregistration_parser.add_argument("hypothesis_id")
    preregistration_parser.add_argument("--dataset", required=True)
    preregistration_parser.add_argument("--metric", required=True)
    preregistration_parser.add_argument("--baseline", required=True)
    split_source = preregistration_parser.add_mutually_exclusive_group(required=True)
    split_source.add_argument(
        "--split-hash",
        help="Frozen dataset split digest as sha256:<64 hexadecimal characters>.",
    )
    split_source.add_argument(
        "--split-file",
        help="Hash a local dataset split without storing its path or contents.",
    )
    preregistration_parser.add_argument("--registered-by", required=True)
    preregistration_parser.add_argument("--minimum-effect", type=float)
    preregistration_parser.add_argument("--alpha", type=float, default=0.05)
    preregistration_parser.add_argument("--minimum-seeds", type=int, default=3)
    preregistration_parser.add_argument("--repo", default=".")
    preregistration_parser.add_argument("-m", "--message")
    preregistration_parser.add_argument("--no-commit", action="store_true")
    preregistration_parser.add_argument("--json", action="store_true", dest="as_json")

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Record one successful, failed, timed-out, or cancelled experiment.",
    )
    experiment_parser.add_argument("summary")
    experiment_parser.add_argument(
        "--status",
        required=True,
        choices=[
            "success",
            "completed",
            "failed",
            "error",
            "timeout",
            "timed_out",
            "cancelled",
            "canceled",
            "running",
        ],
    )
    experiment_parser.add_argument(
        "--study-phase",
        choices=["exploratory", "confirmatory"],
        default="exploratory",
    )
    experiment_parser.add_argument("--plan")
    experiment_parser.add_argument("--preregistration")
    experiment_parser.add_argument(
        "--metric", action="append", default=[], help="Metric as NAME=VALUE."
    )
    experiment_parser.add_argument("--seed", action="append", type=int, default=[])
    experiment_parser.add_argument("--environment-hash")
    experiment_parser.add_argument(
        "--dependency-lock-hash", action="append", default=[]
    )
    experiment_parser.add_argument(
        "--dependency-lock-file", action="append", default=[]
    )
    experiment_parser.add_argument("--dataset-hash", action="append", default=[])
    experiment_parser.add_argument("--code-commit")
    experiment_parser.add_argument("--failure-class", default="")
    experiment_parser.add_argument(
        "--reproduce-command",
        help="Shell-free command stored for later `research reproduce --execute`.",
    )
    experiment_parser.add_argument("--repo", default=".")
    experiment_parser.add_argument("-m", "--message")
    experiment_parser.add_argument("--no-commit", action="store_true")
    experiment_parser.add_argument("--json", action="store_true", dest="as_json")

    evidence_parser = subparsers.add_parser(
        "evidence",
        help="Bind a result to experiment attempts and supported/refuted objects.",
    )
    evidence_parser.add_argument("result")
    evidence_parser.add_argument("--attempt", action="append", required=True)
    evidence_parser.add_argument("--supports", action="append", default=[])
    evidence_parser.add_argument("--refutes", action="append", default=[])
    evidence_parser.add_argument(
        "--metric", action="append", default=[], help="Metric as NAME=VALUE."
    )
    evidence_parser.add_argument("--verified", action="store_true")
    evidence_parser.add_argument(
        "--verifier",
        help="Independent verifier identity; required with --verified.",
    )
    evidence_parser.add_argument("--repo", default=".")
    evidence_parser.add_argument("-m", "--message")
    evidence_parser.add_argument("--no-commit", action="store_true")
    evidence_parser.add_argument("--json", action="store_true", dest="as_json")

    review_parser = subparsers.add_parser(
        "review",
        help="Record an independent review and compute its promotion gate.",
    )
    review_parser.add_argument("summary")
    review_parser.add_argument("--evaluates", action="append", required=True)
    review_parser.add_argument("--verifier", required=True)
    review_parser.add_argument("--decision", choices=["pass", "hold"], required=True)
    review_parser.add_argument("--failure", action="append", default=[])
    review_parser.add_argument("--repo", default=".")
    review_parser.add_argument("-m", "--message")
    review_parser.add_argument("--no-commit", action="store_true")
    review_parser.add_argument("--json", action="store_true", dest="as_json")

    claim_parser = subparsers.add_parser(
        "claim",
        help="Record an evidence-bound claim; verified claims require a passing gate.",
    )
    claim_parser.add_argument("statement")
    claim_parser.add_argument("--evidence", action="append", required=True)
    claim_parser.add_argument("--scope", default="")
    claim_parser.add_argument("--gate")
    claim_parser.add_argument("--verified", action="store_true")
    claim_parser.add_argument("--repo", default=".")
    claim_parser.add_argument("-m", "--message")
    claim_parser.add_argument("--no-commit", action="store_true")
    claim_parser.add_argument("--json", action="store_true", dest="as_json")

    init_parser = subparsers.add_parser(
        "init", help="Initialize a local research repository."
    )
    init_parser.add_argument("path")
    init_parser.add_argument("--name")
    init_parser.add_argument("--question")
    init_parser.add_argument("--question-file")
    init_parser.add_argument(
        "--policy",
        choices=["manual", "stage", "milestone"],
        default="milestone",
    )
    init_parser.add_argument("--actor", default="xscientist")
    init_parser.add_argument("--git-user-name")
    init_parser.add_argument("--git-user-email")
    init_parser.add_argument("--max-file-bytes", type=int, default=2 * 1024 * 1024)
    init_parser.add_argument("--no-commit", action="store_true")
    init_parser.add_argument("--json", action="store_true", dest="as_json")

    status_parser = subparsers.add_parser(
        "status", help="Show research and storage status."
    )
    status_parser.add_argument("--repo", default=".")
    status_parser.add_argument("--json", action="store_true", dest="as_json")

    audit_parser = subparsers.add_parser(
        "audit",
        help="Audit claim-to-evidence-to-reproduction closure without disclosing payloads.",
    )
    audit_parser.add_argument("ref", nargs="?", default="HEAD")
    audit_parser.add_argument("--repo", default=".")
    audit_parser.add_argument(
        "--level", choices=["trace", "replay", "verify"], default="trace"
    )
    audit_parser.add_argument("--no-objects", action="store_true")
    audit_parser.add_argument("--json", action="store_true", dest="as_json")

    decide_parser = subparsers.add_parser(
        "decide",
        help="Explain whether the next research transition should checkpoint, fork, or merge.",
    )
    decide_parser.add_argument(
        "event",
        choices=[
            "observation",
            "hypothesis",
            "preregistration",
            "experiment-started",
            "experiment-completed",
            "experiment-failed",
            "evidence",
            "review",
            "gate",
            "manuscript",
            "release",
            "method-change",
            "contradiction",
            "replication",
            "agent-candidate",
            "merge-candidate",
        ],
    )
    decide_parser.add_argument("--repo", default=".")
    decide_parser.add_argument("--name", default="")
    decide_parser.add_argument("--state", default="")
    decide_parser.add_argument("--source-branch")
    decide_parser.add_argument("--competing-hypothesis", action="store_true")
    decide_parser.add_argument("--contradictory-evidence", action="store_true")
    decide_parser.add_argument("--protocol-change", action="store_true")
    decide_parser.add_argument("--independent-replication", action="store_true")
    decide_parser.add_argument("--json", action="store_true", dest="as_json")

    tree_parser = subparsers.add_parser(
        "tree",
        help="Show the payload-free semantic technology tree and open frontier.",
    )
    tree_parser.add_argument("--repo", default=".")
    tree_parser.add_argument("--json", action="store_true", dest="as_json")

    fsck_parser = subparsers.add_parser(
        "fsck", help="Verify checkpoints, ARA bindings, pointers, and CAS objects."
    )
    fsck_parser.add_argument("--repo", default=".")
    fsck_parser.add_argument("--commit", default="HEAD")
    fsck_parser.add_argument("--no-objects", action="store_true")
    fsck_parser.add_argument("--json", action="store_true", dest="as_json")

    checkpoint_parser = subparsers.add_parser(
        "checkpoint", help="Commit a safe, reproducible scientific checkpoint."
    )
    checkpoint_parser.add_argument("--repo", default=".")
    checkpoint_parser.add_argument("--stage", required=True)
    checkpoint_parser.add_argument("--subject", required=True)
    checkpoint_parser.add_argument("--summary", default="")
    checkpoint_parser.add_argument("--status", default="completed")
    checkpoint_parser.add_argument("--actor")
    checkpoint_parser.add_argument("--node", action="append", default=[])
    checkpoint_parser.add_argument("--claim", action="append", default=[])
    checkpoint_parser.add_argument("--ara", action="append", default=[])
    checkpoint_parser.add_argument("--object-ref", action="append", default=[])
    checkpoint_parser.add_argument("--reproduce")
    checkpoint_parser.add_argument("--include", action="append", default=[])
    checkpoint_parser.add_argument("--no-commit", action="store_true")
    checkpoint_parser.add_argument("--allow-checkpoint-only", action="store_true")
    checkpoint_parser.add_argument(
        "--staged",
        action="store_true",
        help="Commit exactly the paths selected by `research stage`.",
    )
    checkpoint_parser.add_argument("--json", action="store_true", dest="as_json")

    record_parser = subparsers.add_parser(
        "record", help="Record one immutable, typed scientific object."
    )
    record_parser.add_argument("kind", choices=RESEARCH_OBJECT_KINDS)
    record_parser.add_argument("--repo", default=".")
    record_parser.add_argument(
        "--state", choices=RESEARCH_OBJECT_STATES, default="draft"
    )
    record_parser.add_argument("--data", help="Payload as a JSON object.")
    record_parser.add_argument("--file", help="Read the JSON payload from a file.")
    record_parser.add_argument(
        "--relation",
        action="append",
        default=[],
        help="Scientific relation as TYPE:TARGET[:ROLE].",
    )
    record_parser.add_argument("--json", action="store_true", dest="as_json")

    objects_parser = subparsers.add_parser(
        "objects", help="List or inspect typed scientific objects."
    )
    objects_parser.add_argument("object_id", nargs="?")
    objects_parser.add_argument("--repo", default=".")
    objects_parser.add_argument("--kind", choices=RESEARCH_OBJECT_KINDS)
    objects_parser.add_argument("--state", choices=RESEARCH_OBJECT_STATES)
    objects_parser.add_argument("--json", action="store_true", dest="as_json")

    stage_parser = subparsers.add_parser(
        "stage", help="Select exact research changes for the next checkpoint."
    )
    stage_parser.add_argument("paths", nargs="*")
    stage_parser.add_argument("--repo", default=".")
    stage_parser.add_argument("--all", action="store_true", dest="all_changes")
    stage_parser.add_argument("--json", action="store_true", dest="as_json")

    add_parser = subparsers.add_parser(
        "add", help="Git-style alias for selecting exact research changes."
    )
    add_parser.add_argument("paths", nargs="*")
    add_parser.add_argument("--repo", default=".")
    add_parser.add_argument("-A", "--all", action="store_true", dest="all_changes")
    add_parser.add_argument("--json", action="store_true", dest="as_json")

    unstage_parser = subparsers.add_parser(
        "unstage", help="Remove paths from research staging without changing files."
    )
    unstage_parser.add_argument("paths", nargs="*")
    unstage_parser.add_argument("--repo", default=".")
    unstage_parser.add_argument("--all", action="store_true", dest="all_paths")
    unstage_parser.add_argument("--json", action="store_true", dest="as_json")

    commit_parser = subparsers.add_parser(
        "commit", help="Create a checkpoint from the native research stage."
    )
    commit_parser.add_argument("--repo", default=".")
    commit_parser.add_argument("-m", "--message", required=True)
    commit_parser.add_argument("--stage", default="research")
    commit_parser.add_argument("--summary", default="")
    commit_parser.add_argument("--status", default="completed")
    commit_parser.add_argument("--actor")
    commit_parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="all_changes",
        help="Select all eligible research changes before committing.",
    )
    commit_parser.add_argument("--json", action="store_true", dest="as_json")

    branch_parser = subparsers.add_parser(
        "branch", help="List or fork independent research lines."
    )
    branch_parser.add_argument("name", nargs="?")
    branch_parser.add_argument("--repo", default=".")
    branch_parser.add_argument("--from", default="HEAD", dest="from_ref")
    branch_parser.add_argument("--switch", action="store_true")
    branch_action = branch_parser.add_mutually_exclusive_group()
    branch_action.add_argument("-d", "--delete", action="store_true")
    branch_action.add_argument("-D", "--force-delete", action="store_true")
    branch_action.add_argument("-m", "--move", metavar="NEW_NAME")
    branch_parser.add_argument("--json", action="store_true", dest="as_json")

    switch_parser = subparsers.add_parser(
        "switch", help="Switch to another clean research line."
    )
    switch_parser.add_argument("name")
    switch_parser.add_argument("--repo", default=".")
    switch_parser.add_argument("--json", action="store_true", dest="as_json")

    restore_parser = subparsers.add_parser(
        "restore", help="Restore explicit research paths from a checkpoint."
    )
    restore_parser.add_argument("source")
    restore_parser.add_argument("paths", nargs="+")
    restore_parser.add_argument("--repo", default=".")
    restore_parser.add_argument("--json", action="store_true", dest="as_json")

    revert_parser = subparsers.add_parser(
        "revert", help="Revert a checkpoint and record the reversal scientifically."
    )
    revert_parser.add_argument("commit")
    revert_parser.add_argument("--repo", default=".")
    revert_parser.add_argument("-m", "--message")
    revert_parser.add_argument("--json", action="store_true", dest="as_json")

    tag_parser = subparsers.add_parser(
        "tag", help="List or immutably name scientific checkpoints."
    )
    tag_parser.add_argument("name", nargs="?")
    tag_parser.add_argument("--repo", default=".")
    tag_parser.add_argument("--commit", default="HEAD")
    tag_parser.add_argument("--annotation", default="")
    tag_parser.add_argument("--json", action="store_true", dest="as_json")

    blame_parser = subparsers.add_parser(
        "blame", help="Trace a scientific object to its originating checkpoint."
    )
    blame_parser.add_argument("object_id")
    blame_parser.add_argument("--repo", default=".")
    blame_parser.add_argument("--commit", default="HEAD")
    blame_parser.add_argument("--json", action="store_true", dest="as_json")

    merge_parser = subparsers.add_parser(
        "merge", help="Preflight or merge a scientifically compatible research line."
    )
    merge_parser.add_argument("source")
    merge_parser.add_argument("--repo", default=".")
    merge_parser.add_argument("--preview", action="store_true")
    merge_parser.add_argument("--subject")
    merge_parser.add_argument("--summary", default="")
    merge_parser.add_argument("--actor")
    merge_parser.add_argument(
        "--preserve-conflicts",
        action="store_true",
        help=(
            "preserve opposed evidence and add a rejected hold gate; other conflict "
            "types remain blocked"
        ),
    )
    merge_parser.add_argument("--json", action="store_true", dest="as_json")

    log_parser = subparsers.add_parser("log", help="Show scientific history.")
    log_parser.add_argument("--repo", default=".")
    log_parser.add_argument("--limit", type=int, default=20)
    log_parser.add_argument("--json", action="store_true", dest="as_json")

    show_parser = subparsers.add_parser("show", help="Show a checkpoint at a commit.")
    show_parser.add_argument("commit", nargs="?", default="HEAD")
    show_parser.add_argument("--repo", default=".")
    show_parser.add_argument("--json", action="store_true", dest="as_json")

    diff_parser = subparsers.add_parser("diff", help="Compare two research commits.")
    diff_parser.add_argument("before", nargs="?", default="HEAD~1")
    diff_parser.add_argument("after", nargs="?", default="HEAD")
    diff_parser.add_argument("--repo", default=".")
    diff_parser.add_argument(
        "--deep",
        action="store_true",
        help="Compare structured scientific JSON fields up to the safety limit.",
    )
    diff_parser.add_argument("--json", action="store_true", dest="as_json")

    object_parser = subparsers.add_parser(
        "object", help="Register large evidence in the local content-addressed store."
    )
    object_subparsers = object_parser.add_subparsers(
        dest="object_command", required=True
    )
    object_add = object_subparsers.add_parser(
        "add", help="Add a file and write a Git-safe pointer."
    )
    object_add.add_argument("source")
    object_add.add_argument("--repo", default=".")
    object_add.add_argument("--logical-path")
    object_add.add_argument("--media-type")
    object_add.add_argument("--json", action="store_true", dest="as_json")

    bundle_parser = subparsers.add_parser(
        "bundle", help="Create, verify, or restore an offline research bundle."
    )
    bundle_parser.add_argument(
        "action",
        nargs="?",
        choices=["create", "verify", "restore"],
        default="create",
    )
    bundle_parser.add_argument("bundle_path", nargs="?")
    bundle_parser.add_argument("--repo", default=".")
    bundle_parser.add_argument("--dest")
    bundle_parser.add_argument(
        "--profile", choices=["index", "reproduce", "audit"], default="reproduce"
    )
    bundle_parser.add_argument("--allow-incomplete", action="store_true")
    bundle_parser.add_argument("--json", action="store_true", dest="as_json")

    export_parser = subparsers.add_parser(
        "export",
        help="Export one committed research state to standard ecosystem formats.",
    )
    export_parser.add_argument("--repo", default=".")
    export_parser.add_argument("--ref", default="HEAD")
    export_parser.add_argument("--dest", required=True)
    export_parser.add_argument(
        "--format",
        action="append",
        choices=["ro-crate", "prov-json", "cwl", "dvc", "mlflow"],
        default=[],
        dest="formats",
        help="Repeat to select formats; the default exports all adapters.",
    )
    export_parser.add_argument(
        "--include-payloads",
        action="store_true",
        help="Include scientific payloads in RO-Crate; metadata-only is safer by default.",
    )
    export_parser.add_argument("--json", action="store_true", dest="as_json")

    reproduce_parser = subparsers.add_parser(
        "reproduce", help="Inspect or materialize a commit's reproduction closure."
    )
    reproduce_parser.add_argument("commit", nargs="?", default="HEAD")
    reproduce_parser.add_argument("--repo", default=".")
    reproduce_parser.add_argument("--dest")
    reproduce_parser.add_argument("--execute", action="store_true")
    reproduce_parser.add_argument(
        "--record",
        action="store_true",
        help="Record the compact receipt back into Research VCS.",
    )
    reproduce_parser.add_argument(
        "--reproduces",
        action="append",
        default=[],
        help="Typed object ID checked by the receipt; repeat as needed.",
    )
    reproduce_parser.add_argument("--verifier")
    reproduce_parser.add_argument("--verified", action="store_true")
    reproduce_parser.add_argument("--no-commit", action="store_true")
    reproduce_parser.add_argument("--timeout", type=int, default=600)
    reproduce_parser.add_argument(
        "--environment-policy",
        choices=["ignore", "warn", "strict"],
        default="warn",
    )
    reproduce_parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _human_status(payload: dict[str, Any]) -> None:
    print(f"Repository:        {_display_path(payload['repository'])}")
    print(f"Branch:            {payload['branch']}")
    print(f"HEAD:              {payload['head'] or '(no commit)'}")
    print(f"Checkpoint policy: {payload['checkpoint_policy']}")
    print(f"Auto push:         {payload['auto_push']}")
    print(f"Research staged:   {len(payload['research_stage']['paths'])}")
    print(f"Eligible changes:  {len(payload['eligible_changes'])}")
    print(f"Excluded changes:  {len(payload['excluded_changes'])}")
    for path in payload["eligible_changes"]:
        print(f"  eligible:        {path}")
    for path in payload["excluded_changes"]:
        print(f"  excluded:        {_display_text(path)}")
    store = payload["object_store"]
    print(f"Local CAS:         {store['objects']} objects / {store['bytes']} bytes")
    previous = payload.get("last_checkpoint") or {}
    if previous:
        print(
            f"Last checkpoint:   {previous.get('checkpoint_id')} "
            f"({previous.get('stage')} / {previous.get('status')})"
        )


def _object_summary(payload: dict[str, Any]) -> str:
    data = payload.get("payload") or {}
    for key in (
        "statement",
        "summary",
        "result",
        "title",
        "text",
        "decision",
        "status",
        "name",
    ):
        value = data.get(key)
        if value not in (None, "", [], {}):
            compact = " ".join(str(value).split())
            return compact[:77] + ("..." if len(compact) > 77 else "")
    return "(no summary)"


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "xscientist research",
) -> int:
    parser = _build_parser(prog=prog)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .git_support import inspect_git_backend

            payload = inspect_git_backend()
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Research VCS backend: {payload['backend']}")
                print(f"Available:            {payload['available']}")
                print(f"Version:              {payload['version'] or 'N/A'}")
                for name, ready in payload["capabilities"].items():
                    print(f"{name.replace('_', ' ').title():<22} {ready}")
                for error in payload["errors"]:
                    print(f"error: {_display_text(error)}", file=sys.stderr)
                if payload.get("install_hint"):
                    print(_display_text(payload["install_hint"]), file=sys.stderr)
            return 0 if payload["ok"] else 1

        if args.command == "hypothesis":
            from .research_commands import save_hypothesis

            result = save_hypothesis(
                args.repo,
                statement=args.statement,
                falsifier=args.falsifier,
                rationale=args.rationale,
                predictions=args.prediction,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("hypothesis", result, as_json=args.as_json)
            return 0

        if args.command == "experiment":
            from .research_commands import save_experiment

            result = save_experiment(
                args.repo,
                summary=args.summary,
                status=args.status,
                study_phase=args.study_phase,
                plan_id=args.plan,
                preregistration_id=args.preregistration,
                metrics=_parse_assignments(args.metric, label="metric"),
                seeds=args.seed,
                environment_hash=args.environment_hash,
                dependency_lock_hashes=[
                    *args.dependency_lock_hash,
                    *(_hash_local_file(path) for path in args.dependency_lock_file),
                ],
                dataset_hashes=args.dataset_hash,
                code_commit=args.code_commit,
                failure_class=args.failure_class,
                reproduce_command=args.reproduce_command,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("experiment", result, as_json=args.as_json)
            return 0

        if args.command == "preregister":
            from .research_commands import save_preregistration

            result = save_preregistration(
                args.repo,
                hypothesis_id=args.hypothesis_id,
                dataset=args.dataset,
                metric=args.metric,
                baseline=args.baseline,
                split_hash=args.split_hash or _hash_local_file(args.split_file),
                registered_by=args.registered_by,
                minimum_effect=args.minimum_effect,
                alpha=args.alpha,
                minimum_seeds=args.minimum_seeds,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("preregistration", result, as_json=args.as_json)
            return 0

        if args.command == "evidence":
            from .research_commands import save_evidence

            result = save_evidence(
                args.repo,
                result_summary=args.result,
                attempt_ids=args.attempt,
                supports=args.supports,
                refutes=args.refutes,
                metrics=_parse_assignments(args.metric, label="metric"),
                verified=args.verified,
                verifier_id=args.verifier,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("evidence", result, as_json=args.as_json)
            return 0

        if args.command == "claim":
            from .research_commands import save_claim

            result = save_claim(
                args.repo,
                statement=args.statement,
                evidence_ids=args.evidence,
                scope=args.scope,
                gate_id=args.gate,
                verified=args.verified,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("claim", result, as_json=args.as_json)
            return 0

        if args.command == "review":
            from .research_commands import save_review

            result = save_review(
                args.repo,
                summary=args.summary,
                evaluates=args.evaluates,
                verifier_id=args.verifier,
                decision=args.decision,
                required_failures=args.failure,
                message=args.message,
                commit=not args.no_commit,
            )
            _print_saved_object("gate decision", result, as_json=args.as_json)
            return 0

        if args.command == "init":
            result = init_repository(
                args.path,
                name=args.name,
                question=_read_question(args),
                policy=args.policy,
                actor=args.actor,
                git_user_name=args.git_user_name,
                git_user_email=args.git_user_email,
                max_file_bytes=args.max_file_bytes,
                commit=not args.no_commit,
            )
            payload = result.to_dict()
            if args.as_json:
                _print_json(payload)
            else:
                print(
                    "Initialized local research repository: "
                    f"{_display_path(args.path)}"
                )
                if result.commit:
                    print(
                        f"Initial checkpoint: {result.commit[:12]} {result.checkpoint_id}"
                    )
                print("Remote: none (XScientist never pushes automatically)")
            return 0

        if args.command == "status":
            payload = repository_status(args.repo)
            _print_json(payload) if args.as_json else _human_status(payload)
            return 0

        if args.command == "audit":
            from .research_closure import audit_research_closure

            payload = audit_research_closure(
                args.repo,
                ref=args.ref,
                level=args.level,
                verify_objects=not args.no_objects,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Scientific closure: {payload['status']}")
                print(f"Target level:       {payload['target_level']}")
                print(f"Commit:             {payload['commit']}")
                print(f"Claims:             {len(payload['claims'])}")
                print(f"Blockers:           {len(payload['blockers'])}")
                for item in payload["blockers"]:
                    target = f" ({item['object_id']})" if item["object_id"] else ""
                    print(f"  {item['code']}{target}: {_display_text(item['message'])}")
                for item in payload["warnings"]:
                    print(
                        f"warning: {item['code']}: {_display_text(item['message'])}",
                        file=sys.stderr,
                    )
            return 0 if payload["complete"] else 1

        if args.command == "decide":
            from .research_policy import decide_research_transition

            payload = decide_research_transition(
                args.repo,
                event=args.event,
                name=args.name,
                state=args.state,
                source_branch=args.source_branch,
                competing_hypothesis=args.competing_hypothesis,
                contradictory_evidence=args.contradictory_evidence,
                protocol_change=args.protocol_change,
                independent_replication=args.independent_replication,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Decision: {payload['decision_id']}")
                print(f"Event:    {payload['event']}")
                print(f"Branch:   {payload['branch']}")
                for action in payload["actions"]:
                    print(f"{action['action']}: {_display_text(action['reason'])}")
                    for command in action["commands"]:
                        print(f"  {command}")
            return 0

        if args.command == "tree":
            from .research_policy import build_research_technology_tree

            payload = build_research_technology_tree(args.repo)
            if args.as_json:
                _print_json(payload)
            else:
                counts = payload["counts"]
                print(
                    "Technology tree: "
                    f"{counts['nodes']} objects, {counts['edges']} relations, "
                    f"{counts['branches']} research lines"
                )
                print(f"Integrity:       {payload['integrity']['ok']}")
                print(f"Open frontier:   {len(payload['frontier'])}")
                for item in payload["frontier"]:
                    print(
                        f"  {item['object_id']} {item['kind']} "
                        f"[{item['classification']}]"
                    )
            return 0 if payload["integrity"]["ok"] else 1

        if args.command == "fsck":
            payload = verify_research_repository(
                args.repo,
                commit=args.commit,
                verify_objects=not args.no_objects,
            )
            if args.as_json:
                _print_json(payload)
            else:
                verdict = "ok" if payload["ok"] else "failed"
                print(f"Research repository: {verdict}")
                print(f"Commit:              {payload['commit']}")
                checked = payload["checked"]
                print(
                    "Checked:             "
                    f"{checked['checkpoints']} checkpoints, "
                    f"{checked['pointers']} pointers, "
                    f"{checked['objects']} objects, "
                    f"{checked['ara_manifests']} ARA manifests"
                )
                for warning in payload["warnings"]:
                    print(f"warning: {_display_text(warning)}", file=sys.stderr)
                for error in payload["errors"]:
                    print(f"error: {_display_text(error)}", file=sys.stderr)
            return 0 if payload["ok"] else 1

        if args.command == "checkpoint":
            operation = commit_research_stage if args.staged else create_checkpoint
            checkpoint_kwargs = dict(
                repo=args.repo,
                stage=args.stage,
                subject=args.subject,
                summary=args.summary,
                status=args.status,
                actor=args.actor,
                nodes=args.node,
                claims=args.claim,
                ara_paths=args.ara,
                object_refs=args.object_ref,
                reproduce_command=args.reproduce,
            )
            if not args.staged:
                checkpoint_kwargs.update(
                    include=args.include,
                    commit=not args.no_commit,
                    allow_checkpoint_only=args.allow_checkpoint_only,
                )
            elif args.no_commit or args.allow_checkpoint_only or args.include:
                raise ResearchGitError(
                    "--staged cannot be combined with --no-commit, "
                    "--allow-checkpoint-only, or --include"
                )
            result = operation(**checkpoint_kwargs)
            payload = result.to_dict()
            if args.as_json:
                _print_json(payload)
            elif result.committed:
                print(
                    f"Checkpoint committed: {result.commit[:12]} {result.checkpoint_id}"
                )
                print(f"Stage: {args.stage}; files: {len(result.staged_paths)}")
            elif result.created:
                print(f"Checkpoint written: {_display_path(result.checkpoint_path)}")
            else:
                print(f"Checkpoint skipped: {result.reason}")
            if result.excluded_paths and not args.as_json:
                print("Excluded by safety policy:")
                for item in result.excluded_paths:
                    print(f"  - {item}")
            return 0

        if args.command == "record":
            if args.state in {"verified", "promoted"}:
                raise ResearchGitError(
                    "raw record cannot create verified or promoted objects; "
                    "use the evidence/review/claim/reproduce lifecycle commands"
                )
            result = record_research_object(
                args.repo,
                kind=args.kind,
                state=args.state,
                payload=_read_object_payload(args),
                relations=_parse_relations(args.relation),
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                action = "Recorded" if result.created else "Already recorded"
                print(f"{action}: {result.object_id}")
                print(f"Kind/state: {result.kind} / {result.state}")
                print(f"Path:       {_display_path(result.path)}")
            return 0

        if args.command == "objects":
            if args.object_id:
                payload = load_research_object(args.repo, args.object_id)
            else:
                payload = list_research_objects(
                    args.repo,
                    kind=args.kind,
                    state=args.state,
                )
            if args.as_json or args.object_id:
                _print_json(payload)
            else:
                for item in payload:
                    print(
                        f"{item['object_id']} {item['kind']} "
                        f"[{item['state']}] {_display_text(_object_summary(item))}"
                    )
            return 0

        if args.command in {"stage", "add"}:
            result = research_stage(
                args.repo,
                args.paths,
                all_changes=args.all_changes,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Research stage: {len(result.paths)} path(s)")
                for path in result.paths:
                    print(f"  {path}")
                for item in result.excluded:
                    print(f"excluded: {_display_text(item)}", file=sys.stderr)
            return 0

        if args.command == "commit":
            if args.all_changes:
                research_stage(args.repo, all_changes=True)
            result = commit_research_stage(
                args.repo,
                stage=args.stage,
                subject=args.message,
                summary=args.summary,
                status=args.status,
                actor=args.actor,
            )
            if args.as_json:
                _print_json(result.to_dict())
            elif result.committed:
                print(
                    f"Research checkpoint: {result.checkpoint_id} "
                    f"files={len(result.staged_paths)}"
                )
            else:
                print(f"Research checkpoint skipped: {result.reason}")
            return 0

        if args.command == "unstage":
            result = research_unstage(
                args.repo,
                args.paths,
                all_paths=args.all_paths,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Research stage: {len(result.paths)} path(s)")
            return 0

        if args.command == "branch":
            if (args.delete or args.force_delete or args.move) and not args.name:
                raise ResearchGitError("branch maintenance requires a branch name")
            if args.delete or args.force_delete:
                payload = delete_research_branch(
                    args.repo,
                    args.name,
                    force=args.force_delete,
                )
            elif args.move:
                payload = rename_research_branch(args.repo, args.name, args.move)
            elif args.name:
                payload = create_research_branch(
                    args.repo,
                    args.name,
                    from_ref=args.from_ref,
                    switch=args.switch,
                )
            else:
                payload = list_research_branches(args.repo)
            if args.as_json:
                _print_json(payload)
            elif args.delete or args.force_delete:
                print(f"Research branch deleted: {payload['name']}")
            elif args.move:
                print(
                    f"Research branch renamed: {payload['old_name']} -> {payload['name']}"
                )
            elif args.name:
                marker = " and switched" if args.switch else ""
                print(f"Research branch created{marker}: {payload['name']}")
            else:
                for item in payload:
                    marker = "*" if item["current"] else " "
                    print(
                        f"{marker} {item['name']} "
                        f"{(item['checkpoint_id'] or '-')}: {item['subject']}"
                    )
            return 0

        if args.command == "switch":
            payload = switch_research_branch(args.repo, args.name)
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Research branch: {payload['name']}")
                print(f"Commit:          {payload['commit']}")
            return 0

        if args.command == "restore":
            payload = restore_research_paths(args.repo, args.source, args.paths)
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Restored from: {payload['source']}")
                for path in payload["paths"]:
                    print(f"  {path}")
                print("Review the working changes, then checkpoint them explicitly.")
            return 0

        if args.command == "revert":
            payload = revert_research_checkpoint(
                args.repo,
                args.commit,
                subject=args.message,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Reverted commit:    {payload['reverted']}")
                print(f"Backend revert:     {payload['revert_commit']}")
                print(
                    "Research checkpoint: "
                    f"{payload['checkpoint'].get('checkpoint_id') or '(not committed)'}"
                )
            return 0

        if args.command == "tag":
            payload = (
                create_research_tag(
                    args.repo,
                    args.name,
                    commit=args.commit,
                    annotation=args.annotation,
                )
                if args.name
                else list_research_tags(args.repo)
            )
            if args.as_json:
                _print_json(payload)
            elif args.name:
                print(
                    f"Research tag: {payload['name']} -> " f"{payload['checkpoint_id']}"
                )
            else:
                for item in payload:
                    print(f"{item['name']} {item['checkpoint_id'] or '-'}")
            return 0

        if args.command == "blame":
            payload = research_blame(
                args.repo,
                args.object_id,
                commit=args.commit,
            )
            if args.as_json:
                _print_json(payload)
            else:
                origin = payload["origin"]
                research_object = payload["object"]
                print(
                    f"{research_object['object_id']} "
                    f"{research_object['kind']} [{research_object['state']}]"
                )
                print(f"Origin checkpoint: {origin['checkpoint_id'] or '-'}")
                print(f"Origin commit:     {origin['commit']}")
                print(f"Subject:           {origin['subject']}")
                print(f"Outgoing links:    {len(payload['relations'])}")
                print(f"Incoming links:    {len(payload['related_by'])}")
            return 0

        if args.command == "merge":
            if args.preview:
                payload = preview_research_merge(args.repo, args.source)
            else:
                payload = merge_research_branch(
                    args.repo,
                    args.source,
                    subject=args.subject,
                    summary=args.summary,
                    actor=args.actor,
                    preserve_conflicts=args.preserve_conflicts,
                ).to_dict()
            if args.as_json:
                _print_json(payload)
            elif args.preview:
                verdict = "clean" if payload["clean"] else "blocked"
                print(f"Merge preflight: {verdict}")
                for conflict in payload["conflicts"]:
                    print(f"  {conflict['type']}: {_display_text(conflict['message'])}")
            else:
                print(
                    f"Merged research line: {payload['source']} -> {payload['target']}"
                )
                print(f"Checkpoint:           {payload['checkpoint_id']}")
                print(f"Commit:               {payload['commit']}")
                for object_id in payload.get("resolution_objects", []):
                    print(f"Contested hold gate:  {object_id}")
            return 0

        if args.command == "log":
            payload = research_log(args.repo, limit=args.limit)
            if args.as_json:
                _print_json(payload)
            else:
                for entry in payload:
                    stage = (entry["trailers"].get("Research-Stage") or ["-"])[0]
                    print(
                        f"{entry['short_commit']} {entry['authored_at']} "
                        f"[{stage}] {entry['subject']}"
                    )
            return 0

        if args.command == "show":
            payload = show_checkpoint(args.repo, args.commit)
            if args.as_json:
                _print_json(payload)
            else:
                checkpoint = payload["checkpoint"]
                print(f"Commit:     {payload['commit']}")
                print(f"Checkpoint: {checkpoint.get('checkpoint_id')}")
                print(f"Stage:      {checkpoint.get('stage')}")
                print(f"State:      {checkpoint.get('status')}")
                print(f"Subject:    {checkpoint.get('subject')}")
                print(f"Summary:    {checkpoint.get('summary')}")
                if (checkpoint.get("reproduce") or {}).get("command"):
                    print(f"Reproduce:  {checkpoint['reproduce']['command']}")
            return 0

        if args.command == "diff":
            payload = research_diff(
                args.repo,
                args.before,
                args.after,
                deep=args.deep,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"{payload['before']}..{payload['after']}")
                for line in payload["changes"]:
                    print(line)
                if payload["stat"]:
                    print(payload["stat"])
                semantic = payload["semantic"]
                print(
                    "Scientific checkpoints: "
                    f"{semantic['before_checkpoint']['checkpoint_id']} -> "
                    f"{semantic['after_checkpoint']['checkpoint_id']}"
                )
                print(
                    "Claims: "
                    f"+{len(semantic['claims']['added'])} "
                    f"-{len(semantic['claims']['removed'])}; "
                    "nodes: "
                    f"+{len(semantic['nodes']['added'])} "
                    f"-{len(semantic['nodes']['removed'])}; "
                    "objects: "
                    f"+{len(semantic['objects']['added'])} "
                    f"-{len(semantic['objects']['removed'])}"
                )
                typed = semantic["research_objects"]
                print(
                    "Typed research objects: "
                    f"+{len(typed['added'])} "
                    f"-{len(typed['removed'])}; "
                    f"relations +{len(typed['relations']['added'])} "
                    f"-{len(typed['relations']['removed'])}"
                )
                if args.deep:
                    print(
                        "Structured field changes: "
                        f"{len(semantic['structured_changes'])}"
                    )
            return 0

        if args.command == "object" and args.object_command == "add":
            result = add_research_object(
                args.repo,
                args.source,
                logical_path=args.logical_path,
                media_type=args.media_type,
            )
            if args.as_json:
                _print_json(result.to_dict())
            else:
                print(f"Object:  {result.object_hash}")
                print(f"Pointer: {_display_path(result.pointer_path)}")
                print(f"Store:   {_display_path(result.store_path)}")
                print("Run `xscientist research checkpoint` to commit the pointer.")
            return 0

        if args.command == "export":
            from .research_interop import INTEROP_FORMATS, export_research_interop

            payload = export_research_interop(
                args.repo,
                args.dest,
                ref=args.ref,
                formats=args.formats or INTEROP_FORMATS,
                include_payloads=args.include_payloads,
            )
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Research export: {_display_path(payload['destination'])}")
                print(f"Commit:          {payload['repository_commit']}")
                print(f"Checkpoint:      {payload['checkpoint_id']}")
                print(f"Objects:         {payload['object_count']}")
                print(f"Formats:         {', '.join(payload['formats'])}")
                print(f"Export hash:     {payload['export_hash']}")
            return 0

        if args.command == "bundle":
            if args.action == "create":
                if not args.dest:
                    raise ResearchGitError("bundle create requires --dest")
                payload = create_research_bundle(
                    args.repo,
                    args.dest,
                    profile=args.profile,
                    allow_incomplete=args.allow_incomplete,
                )
            elif args.action == "verify":
                if not args.bundle_path:
                    raise ResearchGitError("bundle verify requires a bundle path")
                payload = verify_research_bundle(args.bundle_path)
            else:
                if not args.bundle_path or not args.dest:
                    raise ResearchGitError(
                        "bundle restore requires a bundle path and --dest"
                    )
                payload = restore_research_bundle(args.bundle_path, args.dest)
            if args.as_json:
                _print_json(payload)
            elif args.action == "verify":
                print(f"Bundle:   {_display_path(payload['bundle'])}")
                print(f"Valid:    {payload['ok']}")
                for warning in payload["warnings"]:
                    print(f"warning: {_display_text(warning)}", file=sys.stderr)
                for error in payload["errors"]:
                    print(f"error: {_display_text(error)}", file=sys.stderr)
            elif args.action == "restore":
                print(f"Bundle:     {_display_path(payload['bundle'])}")
                print(f"Repository: {_display_path(payload['repository'])}")
                print(f"HEAD:       {payload['commit']}")
                print(f"Objects:    {payload['objects_restored']}")
            else:
                print(f"Bundle:   {_display_path(payload['destination'])}")
                print(f"Profile:  {payload['profile']}")
                print(f"Complete: {payload['complete']}")
                print(f"HEAD:     {payload['repository_head']}")
            if args.action == "verify":
                return 0 if payload["ok"] else 1
            return 0

        if args.command == "reproduce":
            if not args.record and (
                args.verified or args.reproduces or args.verifier or args.no_commit
            ):
                raise ResearchGitError(
                    "--verified, --reproduces, --verifier, and --no-commit require --record"
                )
            if args.record and not args.reproduces:
                raise ResearchGitError(
                    "--record requires at least one --reproduces object"
                )
            if args.verified and not str(args.verifier or "").strip():
                raise ResearchGitError("--verified requires --verifier")
            payload = reproduce_checkpoint(
                args.repo,
                commit=args.commit,
                destination=args.dest,
                execute=args.execute,
                timeout_seconds=args.timeout,
                environment_policy=args.environment_policy,
            )
            if args.record:
                from .research_lifecycle import ResearchLifecycle

                recorded = ResearchLifecycle(args.repo).reproduction(
                    payload["receipt"],
                    reproduces=args.reproduces,
                    verifier_id=args.verifier,
                    verified=args.verified,
                    commit=not args.no_commit,
                )
                payload["recorded_reproduction"] = {
                    "object": recorded["reproduction"].to_dict(),
                    "checkpoint": (
                        recorded["checkpoint"].to_dict()
                        if recorded["checkpoint"] is not None
                        else None
                    ),
                }
            if args.as_json:
                _print_json(payload)
            else:
                print(f"Commit:           {payload['commit']}")
                print(f"Checkpoint:       {payload['checkpoint']['checkpoint_id']}")
                print(f"Objects complete: {payload['objects_complete']}")
                print(f"Environment:      {payload['environment']['matches']}")
                print(
                    "Command:          "
                    f"{_display_text(payload['command'] or '(not declared)')}"
                )
                if (
                    args.environment_policy == "warn"
                    and payload["environment"]["mismatches"]
                ):
                    for mismatch in payload["environment"]["mismatches"]:
                        print(
                            f"warning: environment mismatch: {mismatch['field']}",
                            file=sys.stderr,
                        )
                if payload.get("worktree"):
                    print(f"Worktree:         {_display_path(payload['worktree'])}")
                if payload.get("executed"):
                    print(f"Return code:      {payload['returncode']}")
                print(f"Receipt:          {payload['receipt']['receipt_id']}")
                if payload.get("receipt_path"):
                    print(f"Receipt path:     {payload['receipt_path']}")
                if payload.get("recorded_reproduction"):
                    print(
                        "Recorded object:   "
                        f"{payload['recorded_reproduction']['object']['object_id']}"
                    )
            return int(payload.get("returncode") or 0) if args.execute else 0
    except ResearchGitError as exc:
        message = redact_sensitive_text(str(exc))
        if getattr(args, "as_json", False):
            _print_json(
                {
                    "schema_version": "xscientist.error.v1",
                    "ok": False,
                    "error": {
                        "category": "research_vcs_error",
                        "command": args.command,
                        "message": message,
                    },
                }
            )
        else:
            print(f"research vcs error: {message}", file=sys.stderr)
        return 2
    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
