"""Standards-oriented exports for one committed Research VCS state."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml

from ai_scientist.protocol.canonical_json import canonical_content_hash

from .research_git import (
    ResearchGitError,
    list_research_objects_at_ref,
    research_log,
    show_checkpoint,
)

INTEROP_SCHEMA = "xscientist.research-interop.v1"
INTEROP_FORMATS = ("ro-crate", "prov-json", "cwl", "dvc", "mlflow")


def _object_urn(object_id: str) -> str:
    return "urn:xscientist:research-object:" + object_id


def _checkpoint_urn(checkpoint_id: str) -> str:
    return "urn:xscientist:research-checkpoint:" + checkpoint_id


def _summary(item: dict[str, Any]) -> str:
    payload = item.get("payload") or {}
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
        value = payload.get(key)
        if value not in (None, "", [], {}):
            return " ".join(str(value).split())
    return f"{item.get('kind')} {item.get('object_id')}"


def _schema_type(kind: str) -> str:
    return {
        "question": "Question",
        "hypothesis": "CreativeWork",
        "preregistration": "DigitalDocument",
        "research_plan": "PlanAction",
        "experiment_attempt": "Action",
        "metric": "PropertyValue",
        "evidence": "Dataset",
        "claim": "Claim",
        "review": "Review",
        "gate_decision": "ChooseAction",
        "manuscript": "ScholarlyArticle",
        "reproduction": "Action",
        "agent_candidate": "SoftwareSourceCode",
        "agent_evaluation": "Review",
    }.get(kind, "CreativeWork")


def _relation_property(relation_type: str) -> str:
    return {
        "depends_on": "prov:used",
        "derived_from": "prov:wasDerivedFrom",
        "supports": "xscientist:supports",
        "refutes": "xscientist:refutes",
        "supersedes": "prov:wasRevisionOf",
        "reproduces": "xscientist:reproduces",
        "contradicts": "xscientist:contradicts",
        "evaluates": "schema:reviewedItem",
        "promotes": "xscientist:promotes",
    }.get(relation_type, "schema:isRelatedTo")


def build_ro_crate(
    objects: Iterable[dict[str, Any]],
    *,
    checkpoint: dict[str, Any],
    include_payloads: bool = False,
) -> dict[str, Any]:
    rows = sorted(objects, key=lambda item: str(item["object_id"]))
    checkpoint_id = str(checkpoint.get("checkpoint_id") or "unknown")
    root = {
        "@id": "./",
        "@type": "Dataset",
        "name": str(checkpoint.get("subject") or "XScientist research export"),
        "description": str(checkpoint.get("summary") or ""),
        "dateCreated": checkpoint.get("created_at"),
        "hasPart": [{"@id": _object_urn(str(item["object_id"]))} for item in rows],
        "mentions": {"@id": _checkpoint_urn(checkpoint_id)},
    }
    graph: list[dict[str, Any]] = [
        {
            "@id": "ro-crate-metadata.json",
            "@type": "CreativeWork",
            "about": {"@id": "./"},
            "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
        },
        root,
        {
            "@id": _checkpoint_urn(checkpoint_id),
            "@type": "CreateAction",
            "name": str(checkpoint.get("subject") or checkpoint_id),
            "startTime": checkpoint.get("created_at"),
            "actionStatus": str(checkpoint.get("status") or "completed"),
            "result": [{"@id": _object_urn(str(item["object_id"]))} for item in rows],
        },
    ]
    agents: dict[str, dict[str, Any]] = {}
    for item in rows:
        actor = item.get("actor") or {}
        actor_id = str(actor.get("actor_id") or "xscientist")
        agent_urn = "urn:xscientist:actor:" + actor_id
        agents[agent_urn] = {
            "@id": agent_urn,
            "@type": (
                "Person" if actor.get("authority") == "human" else "SoftwareApplication"
            ),
            "name": actor_id,
            "additionalType": str(actor.get("authority") or "unknown"),
        }
        entity: dict[str, Any] = {
            "@id": _object_urn(str(item["object_id"])),
            "@type": _schema_type(str(item.get("kind") or "")),
            "name": _summary(item),
            "additionalType": "xscientist:" + str(item.get("kind") or "object"),
            "dateCreated": item.get("created_at"),
            "creativeWorkStatus": item.get("state"),
            "sha256": str(item.get("content_hash") or "").removeprefix("sha256:"),
            "creator": {"@id": agent_urn},
        }
        for relation in item.get("relations") or []:
            prop = _relation_property(str(relation.get("type") or ""))
            value = {"@id": _object_urn(str(relation.get("target") or ""))}
            existing = entity.get(prop)
            if existing is None:
                entity[prop] = [value]
            else:
                existing.append(value)
        if include_payloads:
            entity["xscientist:payload"] = deepcopy(item.get("payload") or {})
            entity["xscientist:provenance"] = deepcopy(item.get("provenance") or {})
        graph.append(entity)
    graph.extend(agents[key] for key in sorted(agents))
    return {
        "@context": [
            "https://w3id.org/ro/crate/1.1/context",
            {
                "prov": "http://www.w3.org/ns/prov#",
                "schema": "https://schema.org/",
                "xscientist": "https://xscientist.io/ns/",
            },
        ],
        "@graph": graph,
    }


def build_prov_json(
    objects: Iterable[dict[str, Any]], *, checkpoint: dict[str, Any]
) -> dict[str, Any]:
    rows = sorted(objects, key=lambda item: str(item["object_id"]))
    entities: dict[str, Any] = {}
    agents: dict[str, Any] = {}
    derivations: dict[str, Any] = {}
    attributions: dict[str, Any] = {}
    usages: dict[str, Any] = {}
    checkpoint_id = str(checkpoint.get("checkpoint_id") or "unknown")
    activity_id = "xsc:checkpoint-" + checkpoint_id
    for item in rows:
        object_id = str(item["object_id"])
        entity_id = "xsc:" + object_id
        actor_id = str((item.get("actor") or {}).get("actor_id") or "xscientist")
        agent_id = "xsc:actor-" + re.sub(r"[^A-Za-z0-9_.-]", "-", actor_id)
        entities[entity_id] = {
            "prov:type": "xsc:" + str(item.get("kind") or "object"),
            "prov:label": _summary(item),
            "xsc:state": item.get("state"),
            "xsc:contentHash": item.get("content_hash"),
        }
        agents[agent_id] = {
            "prov:label": actor_id,
            "xsc:authority": (item.get("actor") or {}).get("authority"),
        }
        attributions[f"xsc:attribution-{object_id}"] = {
            "prov:entity": entity_id,
            "prov:agent": agent_id,
        }
        for index, relation in enumerate(item.get("relations") or []):
            target = "xsc:" + str(relation.get("target") or "")
            relation_type = str(relation.get("type") or "")
            relation_id = f"xsc:relation-{object_id}-{index}"
            if relation_type == "depends_on":
                usages[relation_id] = {
                    "prov:activity": activity_id,
                    "prov:entity": target,
                    "xsc:role": relation.get("role"),
                }
            else:
                derivations[relation_id] = {
                    "prov:generatedEntity": entity_id,
                    "prov:usedEntity": target,
                    "xsc:relationType": relation_type,
                }
    return {
        "prefix": {
            "prov": "http://www.w3.org/ns/prov#",
            "xsc": "https://xscientist.io/ns/",
        },
        "entity": entities,
        "activity": {
            activity_id: {
                "prov:label": checkpoint.get("subject"),
                "prov:startTime": checkpoint.get("created_at"),
                "xsc:status": checkpoint.get("status"),
            }
        },
        "agent": agents,
        "wasDerivedFrom": derivations,
        "wasAttributedTo": attributions,
        "used": usages,
    }


def _safe_cwl_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]", "-", value).strip("-")
    return normalized or "step"


def _reproduction_commands(
    repo: str | Path, *, ref: str = "HEAD", limit: int = 1_000
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in research_log(repo, limit=limit, ref=ref):
        shown = show_checkpoint(repo, entry["commit"])
        checkpoint = shown["checkpoint"]
        command = str((checkpoint.get("reproduce") or {}).get("command") or "").strip()
        if not command:
            continue
        checkpoint_id = str(checkpoint.get("checkpoint_id") or entry["commit"])
        if checkpoint_id in seen:
            continue
        seen.add(checkpoint_id)
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            raise ResearchGitError(
                f"checkpoint reproduction command is not shell-free: {entry['commit']}"
            ) from exc
        if not argv:
            continue
        commands.append(
            {
                "checkpoint_id": checkpoint_id,
                "commit": entry["commit"],
                "subject": checkpoint.get("subject"),
                "argv": argv,
            }
        )
    return commands


def build_cwl_export(repo: str | Path, *, ref: str = "HEAD") -> dict[str, Any]:
    commands = _reproduction_commands(repo, ref=ref)
    tools: list[dict[str, Any]] = []
    steps: dict[str, Any] = {}
    for index, item in enumerate(commands):
        identifier = _safe_cwl_id(str(item["checkpoint_id"]))
        tool_id = "#tool-" + identifier
        step_id = f"step_{index + 1}_{identifier}"
        tools.append(
            {
                "id": tool_id,
                "class": "CommandLineTool",
                "cwlVersion": "v1.2",
                "label": item.get("subject"),
                "baseCommand": item["argv"],
                "inputs": [],
                "outputs": [],
                "hints": {
                    "xscientist:ResearchCheckpoint": {
                        "checkpoint_id": item["checkpoint_id"],
                        "commit": item["commit"],
                    }
                },
            }
        )
        steps[step_id] = {"run": tool_id, "in": [], "out": []}
    workflow = {
        "id": "#research-workflow",
        "class": "Workflow",
        "cwlVersion": "v1.2",
        "label": "XScientist reproduction checkpoints",
        "inputs": [],
        "outputs": [],
        "steps": steps,
    }
    return {
        "cwlVersion": "v1.2",
        "$namespaces": {"xscientist": "https://xscientist.io/ns/"},
        "$graph": [workflow, *tools],
    }


def build_dvc_export(repo: str | Path, *, ref: str = "HEAD") -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for item in _reproduction_commands(repo, ref=ref):
        name = _safe_cwl_id(str(item["checkpoint_id"])).replace(".", "-")
        stages[name] = {
            "cmd": shlex.join(item["argv"]),
            "meta": {
                "xscientist_checkpoint_id": item["checkpoint_id"],
                "xscientist_commit": item["commit"],
                "subject": item.get("subject"),
            },
        }
    return {"stages": stages}


def build_mlflow_export(objects: Iterable[dict[str, Any]]) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for item in objects:
        if item.get("kind") != "experiment_attempt":
            continue
        payload = item.get("payload") or {}
        provenance = item.get("provenance") or {}
        metrics = {
            str(name): float(value)
            for name, value in (payload.get("metrics") or {}).items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        runs.append(
            {
                "run_name": str(item["object_id"]),
                "status": item.get("state"),
                "tags": {
                    "xscientist.object_id": item["object_id"],
                    "xscientist.content_hash": item.get("content_hash"),
                    "xscientist.study_phase": payload.get("study_phase"),
                    "xscientist.code_commit": provenance.get("code_commit"),
                },
                "params": {
                    "seeds": provenance.get("seeds") or payload.get("seeds") or [],
                    "dataset_hashes": provenance.get("dataset_hashes") or [],
                    "dependency_lock_hashes": provenance.get("dependency_lock_hashes")
                    or [],
                },
                "metrics": metrics,
            }
        )
    return {"schema_version": "xscientist.mlflow-export.v1", "runs": runs}


def _file_hash(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def export_research_interop(
    repo: str | Path,
    destination: str | Path,
    *,
    ref: str = "HEAD",
    formats: Iterable[str] = INTEROP_FORMATS,
    include_payloads: bool = False,
) -> dict[str, Any]:
    """Atomically export a committed state without overwriting a destination."""

    selected = tuple(dict.fromkeys(str(item).strip() for item in formats))
    unsupported = sorted(set(selected) - set(INTEROP_FORMATS))
    if not selected or unsupported:
        raise ResearchGitError(
            "unsupported interop formats: " + ", ".join(unsupported or ["none"])
        )
    output = Path(destination).expanduser().resolve()
    if output.exists():
        raise ResearchGitError("interop export destination already exists")
    shown = show_checkpoint(repo, ref)
    checkpoint = shown["checkpoint"]
    objects = list_research_objects_at_ref(repo, shown["commit"])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".research-export-", dir=str(output.parent)))
    files: list[Path] = []
    try:
        if "ro-crate" in selected:
            path = staging / "ro-crate-metadata.json"
            path.write_text(
                json.dumps(
                    build_ro_crate(
                        objects,
                        checkpoint=checkpoint,
                        include_payloads=include_payloads,
                    ),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            files.append(path)
        if "prov-json" in selected:
            path = staging / "research.prov.json"
            path.write_text(
                json.dumps(
                    build_prov_json(objects, checkpoint=checkpoint),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            files.append(path)
        if "cwl" in selected:
            path = staging / "research-workflow.cwl"
            path.write_text(
                yaml.safe_dump(
                    build_cwl_export(repo, ref=shown["commit"]),
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            files.append(path)
        if "dvc" in selected:
            path = staging / "dvc.yaml"
            path.write_text(
                yaml.safe_dump(
                    build_dvc_export(repo, ref=shown["commit"]),
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
            files.append(path)
        if "mlflow" in selected:
            path = staging / "mlflow-runs.json"
            path.write_text(
                json.dumps(
                    build_mlflow_export(objects),
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            files.append(path)
        manifest = {
            "schema_version": INTEROP_SCHEMA,
            "repository_commit": shown["commit"],
            "checkpoint_id": checkpoint.get("checkpoint_id"),
            "generated_at": checkpoint.get("created_at"),
            "formats": list(selected),
            "payloads_included": bool(include_payloads),
            "object_count": len(objects),
            "files": [
                {
                    "path": path.name,
                    "hash": _file_hash(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(files)
            ],
        }
        manifest["export_hash"] = canonical_content_hash(manifest)
        manifest_path = staging / "xscientist-export.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
    finally:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
    return {**manifest, "destination": str(output)}


__all__ = [
    "INTEROP_FORMATS",
    "INTEROP_SCHEMA",
    "build_cwl_export",
    "build_dvc_export",
    "build_mlflow_export",
    "build_prov_json",
    "build_ro_crate",
    "export_research_interop",
]
