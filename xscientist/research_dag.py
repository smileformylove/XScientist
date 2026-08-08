"""Unified scientific-evidence DAG projection and offline browser."""

from __future__ import annotations

import html
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from jsonschema import ValidationError, validate as validate_json

from ai_scientist.protocol.canonical_json import canonical_content_hash
from ai_scientist.protocol.graph import analyze_exploration_graph
from ai_scientist.protocol.schemas import load_schema
from ai_scientist.utils.atomic_io import atomic_write_json, atomic_write_text

from .research_closure import audit_research_closure
from .research_git import (
    ResearchGitError,
    list_research_objects_at_ref,
    show_checkpoint,
)

RESEARCH_DAG_SCHEMA = "xscientist.research-dag.v1"

_PHASES = {
    "question": 0,
    "hypothesis": 1,
    "research_plan": 2,
    "preregistration": 2,
    "agent_candidate": 2,
    "experiment_attempt": 3,
    "agent_evaluation": 3,
    "experiment_node": 3,
    "metric": 4,
    "evidence": 4,
    "review": 5,
    "gate_decision": 5,
    "claim": 6,
    "reproduction": 7,
    "manuscript": 8,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_sha256(value: Any) -> bool:
    return bool(re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")))


def _summary(item: Mapping[str, Any], *, disclose: bool) -> str:
    if not disclose:
        return f"{item.get('kind')} {item.get('object_id')}"
    payload = item.get("payload") or {}
    for key in (
        "statement",
        "question",
        "summary",
        "result",
        "title",
        "decision",
        "status",
        "name",
    ):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            compact = " ".join(str(value).split())
            return compact[:157] + ("..." if len(compact) > 157 else "")
    return f"{item.get('kind')} {item.get('object_id')}"


def _has_hash_anchor(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key.endswith("_hash") and _is_sha256(value):
                return True
            if (
                key.endswith("_hashes")
                and isinstance(value, list)
                and bool(value)
                and all(_is_sha256(item) for item in value)
            ):
                return True
            if _has_hash_anchor(value):
                return True
    elif isinstance(payload, list):
        return any(_has_hash_anchor(item) for item in payload)
    return False


def _check(code: str, label: str, passed: bool, layer: str) -> dict[str, Any]:
    return {"code": code, "label": label, "passed": bool(passed), "layer": layer}


def _targets(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    kinds: Iterable[str] = (),
    relations: Iterable[str] = (),
) -> list[str]:
    selected_kinds = set(kinds)
    selected_relations = set(relations)
    return sorted(
        {
            str(relation.get("target"))
            for relation in item.get("relations") or []
            if (
                (not selected_relations or relation.get("type") in selected_relations)
                and str(relation.get("target")) in objects
                and (
                    not selected_kinds
                    or objects[str(relation.get("target"))].get("kind")
                    in selected_kinds
                )
            )
        }
    )


def _object_proof(
    item: Mapping[str, Any],
    objects: Mapping[str, Mapping[str, Any]],
    *,
    claim_closure: Mapping[str, Mapping[str, Any]],
    challenge_targets: set[str],
    closure_blockers: Mapping[str, list[str]],
    closure_warnings: Mapping[str, list[str]],
) -> dict[str, Any]:
    object_id = str(item["object_id"])
    kind = str(item["kind"])
    state = str(item["state"])
    payload = item.get("payload") or {}
    provenance = item.get("provenance") or {}
    actor = item.get("actor") or {}
    checks = [
        _check(
            "content_addressed", "Content hash and object ID validate", True, "trace"
        )
    ]

    if kind == "experiment_attempt":
        checks.extend(
            [
                _check(
                    "plan_bound",
                    "Experiment is bound to a research plan",
                    bool(_targets(item, objects, kinds={"research_plan"})),
                    "trace",
                ),
                _check(
                    "environment_bound",
                    "Environment identity is recorded",
                    bool(provenance.get("environment_hash")),
                    "replay",
                ),
                _check(
                    "dependencies_bound",
                    "Dependency lock identity is recorded",
                    bool(provenance.get("dependency_lock_hashes")),
                    "replay",
                ),
                _check(
                    "code_bound",
                    "Code identity is recorded",
                    bool(
                        provenance.get("code_hash")
                        or provenance.get("code_commit")
                        or payload.get("code_ref")
                    ),
                    "replay",
                ),
                _check(
                    "data_bound",
                    "Dataset identity is recorded",
                    bool(provenance.get("dataset_hashes")),
                    "replay",
                ),
            ]
        )
    elif kind == "evidence":
        checks.extend(
            [
                _check(
                    "attempt_bound",
                    "Evidence is derived from a recorded attempt",
                    bool(
                        _targets(
                            item,
                            objects,
                            kinds={"experiment_attempt"},
                            relations={"derived_from", "depends_on"},
                        )
                    ),
                    "trace",
                ),
                _check(
                    "measurement_bound",
                    "Evidence has an immutable measurement anchor",
                    _has_hash_anchor(payload),
                    "replay",
                ),
                _check(
                    "independently_verified",
                    "Independent verifier accepted the evidence",
                    state == "verified"
                    and actor.get("authority") == "independent_evaluator",
                    "verify",
                ),
            ]
        )
    elif kind == "review":
        checks.append(
            _check(
                "independent_authority",
                "Review was produced by an independent evaluator",
                state == "verified"
                and actor.get("authority") == "independent_evaluator",
                "verify",
            )
        )
    elif kind == "gate_decision":
        checks.append(
            _check(
                "deterministic_authority",
                "Gate was produced by deterministic policy",
                state in {"verified", "promoted"}
                and actor.get("authority") == "deterministic_gate",
                "verify",
            )
        )
        deployment = payload.get("deployment_receipt")
        if isinstance(deployment, Mapping):
            from ai_scientist.utils.evolution_deployment import (
                validate_deployment_receipt,
            )

            checks.append(
                _check(
                    "deployment_receipt",
                    "Production deployment receipt validates",
                    validate_deployment_receipt(deployment)["ok"],
                    "verify",
                )
            )
    elif kind == "reproduction":
        checks.extend(
            [
                _check(
                    "receipt_bound",
                    "Reproduction contains a content-addressed receipt",
                    _has_hash_anchor(payload),
                    "replay",
                ),
                _check(
                    "reproduction_verified",
                    "Independent reproduction passed",
                    state == "verified",
                    "verify",
                ),
            ]
        )
    elif kind == "agent_candidate":
        candidate = payload.get("candidate") or payload
        checks.append(
            _check(
                "candidate_artifact_bound",
                "Evolution candidate is bound to an immutable artifact",
                _is_sha256(candidate.get("candidate_artifact_hash"))
                or _is_sha256(candidate.get("candidate_hash")),
                "replay",
            )
        )
    elif kind == "agent_evaluation":
        checks.append(
            _check(
                "independent_evaluation",
                "Evolution evaluation has independent authority",
                state == "verified"
                and actor.get("authority") == "independent_evaluator",
                "verify",
            )
        )
    elif kind == "claim":
        closure = claim_closure.get(object_id)
        checks.extend(
            [
                _check(
                    "claim_trace",
                    "Claim is linked to evidence, attempt, and plan",
                    bool(closure and closure.get("trace_complete")),
                    "trace",
                ),
                _check(
                    "claim_replay",
                    "Claim lineage has sufficient replay identities",
                    bool(closure and closure.get("replay_ready")),
                    "replay",
                ),
                _check(
                    "claim_verify",
                    "Claim has independent gate and reproduction",
                    bool(closure and closure.get("verified")),
                    "verify",
                ),
            ]
        )

    layer_pass = {
        layer: all(check["passed"] for check in checks if check["layer"] == layer)
        for layer in ("trace", "replay", "verify")
    }
    contested = object_id in challenge_targets
    if contested:
        level = "contested"
    elif (
        checks
        and layer_pass["verify"]
        and any(check["layer"] == "verify" for check in checks)
    ):
        level = "verified"
    elif (
        layer_pass["trace"]
        and layer_pass["replay"]
        and any(check["layer"] == "replay" for check in checks)
    ):
        level = "replayable"
    elif layer_pass["trace"]:
        level = "traceable"
    else:
        level = "recorded"
    blockers = [check["code"] for check in checks if not check["passed"]]
    blockers.extend(closure_blockers.get(object_id, []))
    return {
        "level": level,
        "checks": checks,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(closure_warnings.get(object_id, []))),
        "contested": contested,
    }


def _read_ara(
    root: Path,
    *,
    index: int,
    disclose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    graph_path = root / "exploration_graph.json"
    if not graph_path.is_file():
        raise ResearchGitError(f"ARA exploration graph not found: {root}")
    try:
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchGitError(f"ARA exploration graph is invalid: {root}") from exc
    if not isinstance(graph, dict):
        raise ResearchGitError(f"ARA exploration graph is invalid: {root}")
    analysis = analyze_exploration_graph(graph)
    prefix = f"ara:{index}:"
    nodes: list[dict[str, Any]] = []
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping) or not str(raw.get("id") or ""):
            continue
        raw_id = str(raw["id"])
        summary = (
            " ".join(str(raw.get("analysis") or raw.get("stage") or raw_id).split())
            if disclose
            else "experiment node"
        )
        node_hash = raw.get("content_hash")
        valid_node_hash = _is_sha256(node_hash)
        isolated = bool((raw.get("execution_isolation") or {}).get("isolated"))
        completed = not bool(raw.get("is_buggy"))
        nodes.append(
            {
                "id": prefix + raw_id,
                "source": "ara",
                "kind": "experiment_node",
                "state": "completed" if completed else "failed",
                "phase": _PHASES["experiment_node"],
                "summary": summary[:160],
                "content_hash": str(node_hash) if valid_node_hash else None,
                "actor": None,
                "created_at": None,
                "proof": {
                    "level": (
                        "replayable"
                        if valid_node_hash and isolated
                        else "traceable" if valid_node_hash else "recorded"
                    ),
                    "checks": [
                        _check(
                            "node_content_addressed",
                            "Experiment node has a content hash",
                            valid_node_hash,
                            "trace",
                        ),
                        _check(
                            "execution_isolated",
                            "Experiment executed in an isolated backend",
                            isolated,
                            "replay",
                        ),
                    ],
                    "blockers": [
                        code
                        for code, passed in (
                            ("node_content_addressed", valid_node_hash),
                            ("execution_isolated", isolated),
                        )
                        if not passed
                    ],
                    "warnings": [],
                    "contested": False,
                },
                "source_ref": f"ara:{index}",
            }
        )
    valid_ids = {node["id"] for node in nodes}
    edges = []
    raw_edges = graph.get("edges") or []
    seen: set[tuple[str, str]] = set()
    for raw in raw_edges:
        if not isinstance(raw, Mapping):
            continue
        source, target = prefix + str(raw.get("parent")), prefix + str(raw.get("child"))
        if source in valid_ids and target in valid_ids:
            seen.add((source, target))
    for raw in graph.get("nodes") or []:
        if not isinstance(raw, Mapping):
            continue
        child = prefix + str(raw.get("id"))
        parent_id = raw.get("parent_id")
        if parent_id:
            parent = prefix + str(parent_id)
            if parent in valid_ids and child in valid_ids:
                seen.add((parent, child))
    for source, target in sorted(seen):
        edges.append(
            {
                "source": source,
                "target": target,
                "type": "evolves_to",
                "role": "experiment exploration",
                "category": "evolution",
            }
        )
    manifest_hash = None
    lock_path = root / "manifest.lock"
    if lock_path.is_file():
        try:
            manifest_hash = json.loads(lock_path.read_text(encoding="utf-8")).get(
                "manifest_hash"
            )
        except (OSError, json.JSONDecodeError, AttributeError):
            manifest_hash = None
    if not _is_sha256(manifest_hash):
        manifest_hash = None
    roots = [prefix + node_id for node_id in analysis.get("root_ids") or []]
    leaves = [prefix + node_id for node_id in analysis.get("leaf_ids") or []]
    return (
        nodes,
        edges,
        {
            "name": f"ara:{index}",
            "manifest_hash": manifest_hash,
            "root_ids": roots,
            "leaf_ids": leaves,
            "integrity": analysis,
        },
    )


def build_research_dag(
    repo: str | Path,
    *,
    ref: str = "HEAD",
    ara_roots: Sequence[str | Path] = (),
    disclose_summaries: bool = True,
) -> dict[str, Any]:
    """Project Research VCS and optional ARA execution trees into one DAG."""

    checkpoint = show_checkpoint(repo, ref)
    commit = str(checkpoint["commit"])
    rows = list_research_objects_at_ref(repo, commit)
    objects = {str(item["object_id"]): item for item in rows}
    claims = [item for item in rows if item.get("kind") == "claim"]
    closure = (
        audit_research_closure(repo, ref=commit, level="verify") if claims else None
    )
    claim_closure = {
        str(item["claim_id"]): item for item in (closure or {}).get("claims") or []
    }
    blocker_map: dict[str, list[str]] = defaultdict(list)
    warning_map: dict[str, list[str]] = defaultdict(list)
    for item in (closure or {}).get("blockers") or []:
        blocker_map[str(item.get("object_id") or "")].append(str(item["code"]))
    for item in (closure or {}).get("warnings") or []:
        warning_map[str(item.get("object_id") or "")].append(str(item["code"]))
    challenge_targets = {
        str(relation.get("target"))
        for item in rows
        for relation in item.get("relations") or []
        if relation.get("type") in {"refutes", "contradicts"}
    }
    nodes = [
        {
            "id": str(item["object_id"]),
            "source": "research_vcs",
            "kind": str(item["kind"]),
            "state": str(item["state"]),
            "phase": _PHASES.get(str(item["kind"]), 4),
            "summary": _summary(item, disclose=disclose_summaries),
            "content_hash": str(item["content_hash"]),
            "actor": dict(item.get("actor") or {}),
            "created_at": str(item.get("created_at") or ""),
            "proof": _object_proof(
                item,
                objects,
                claim_closure=claim_closure,
                challenge_targets=challenge_targets,
                closure_blockers=blocker_map,
                closure_warnings=warning_map,
            ),
            "source_ref": commit,
        }
        for item in rows
    ]
    edges: list[dict[str, Any]] = []
    dangling: list[str] = []
    category = {
        "supports": "support",
        "refutes": "challenge",
        "contradicts": "challenge",
        "evaluates": "verification",
        "reproduces": "verification",
        "promotes": "evolution",
        "supersedes": "evolution",
        "depends_on": "lineage",
        "derived_from": "lineage",
    }
    for item in rows:
        for relation in item.get("relations") or []:
            target = str(relation.get("target") or "")
            if target not in objects:
                dangling.append(f"{item['object_id']}:{target}")
                continue
            relation_type = str(relation.get("type") or "depends_on")
            edges.append(
                {
                    "source": target,
                    "target": str(item["object_id"]),
                    "type": relation_type,
                    "role": str(relation.get("role") or ""),
                    "category": category.get(relation_type, "lineage"),
                }
            )

    sources: list[dict[str, Any]] = [
        {"name": "research_vcs", "commit": commit, "object_count": len(rows)}
    ]
    ara_metadata: list[dict[str, Any]] = []
    for index, raw_root in enumerate(ara_roots):
        ara_nodes, ara_edges, metadata_row = _read_ara(
            Path(raw_root).expanduser().resolve(),
            index=index,
            disclose=disclose_summaries,
        )
        nodes.extend(ara_nodes)
        edges.extend(ara_edges)
        ara_metadata.append(metadata_row)
        sources.append(
            {
                "name": metadata_row["name"],
                "manifest_hash": metadata_row["manifest_hash"],
                "object_count": len(ara_nodes),
                "integrity": {
                    "is_dag": metadata_row["integrity"]["is_dag"],
                    "error_count": metadata_row["integrity"]["error_count"],
                    "warning_count": metadata_row["integrity"]["warning_count"],
                },
            }
        )
    for ara in ara_metadata:
        manifest_hash = ara.get("manifest_hash")
        if not manifest_hash:
            continue
        anchored_objects = [
            object_id
            for object_id, item in objects.items()
            if (item.get("provenance") or {}).get("ara_manifest_hash") == manifest_hash
        ]
        for leaf in ara["leaf_ids"]:
            for object_id in anchored_objects:
                edges.append(
                    {
                        "source": leaf,
                        "target": object_id,
                        "type": "anchors",
                        "role": "ARA manifest evidence",
                        "category": "verification",
                    }
                )

    graph_for_analysis = {
        "nodes": [{"id": node["id"]} for node in nodes],
        "edges": [
            {"parent": edge["source"], "child": edge["target"]} for edge in edges
        ],
    }
    analysis = analyze_exploration_graph(graph_for_analysis)
    for ara in ara_metadata:
        source_integrity = ara["integrity"]
        for issue in source_integrity.get("issues") or []:
            copied = dict(issue)
            copied["code"] = f"{ara['name']}_{copied.get('code', 'invalid')}"
            copied["path"] = f"{ara['name']}:{copied.get('path', '')}"
            analysis["issues"].append(copied)
            if copied.get("severity") == "error":
                analysis["error_count"] += 1
            elif copied.get("severity") == "warning":
                analysis["warning_count"] += 1
        if source_integrity.get("error_count"):
            analysis["is_dag"] = False
    if dangling:
        analysis["is_dag"] = False
        analysis["error_count"] += len(dangling)
        analysis["issues"].extend(
            {
                "severity": "error",
                "code": "dangling_research_relation",
                "message": value,
                "path": "edges",
            }
            for value in sorted(dangling)
        )
    proof_counts = Counter(node["proof"]["level"] for node in nodes)
    base = {
        "schema_version": RESEARCH_DAG_SCHEMA,
        "ref": ref,
        "commit": commit,
        "content_disclosed": bool(disclose_summaries),
        "sources": sources,
        "nodes": sorted(nodes, key=lambda node: (node["phase"], node["id"])),
        "edges": sorted(
            edges,
            key=lambda edge: (
                edge["source"],
                edge["target"],
                edge["type"],
                edge["role"],
            ),
        ),
        "proof_summary": dict(sorted(proof_counts.items())),
        "scientific_closure": {
            "status": (closure or {}).get("status", "not_applicable"),
            "claim_count": len(claims),
            "blocker_count": len((closure or {}).get("blockers") or []),
            "warning_count": len((closure or {}).get("warnings") or []),
            "content_hash": (closure or {}).get("content_hash"),
        },
        "integrity": analysis,
    }
    graph = {
        **base,
        "generated_at": _now_iso(),
        "graph_hash": canonical_content_hash(base),
    }
    try:
        validate_json(graph, load_schema("research_dag"))
    except ValidationError as exc:  # pragma: no cover - implementation contract
        raise ResearchGitError(
            f"generated research DAG is invalid: {exc.message}"
        ) from exc
    return graph


def render_research_dag_html(
    graph: Mapping[str, Any],
    *,
    title: str = "XScientist Scientific Evidence DAG",
) -> str:
    """Render a self-contained, searchable evidence and evolution browser."""

    payload = json.dumps(graph, ensure_ascii=False, separators=(",", ":"), default=str)
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e")
    title_text = html.escape(title)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title_text}</title><style>
:root{{color-scheme:light dark;--bg:light-dark(#f7f8fb,#10131a);--panel:light-dark(#fff,#181d27);--ink:light-dark(#18202f,#eef2f8);--muted:light-dark(#667085,#9aa4b5);--line:light-dark(#c9d0dc,#3a4353);--support:#16855b;--challenge:#d14b43;--verify:#7657c8;--evolve:#2f70c9;--warn:#c07818}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}}header{{padding:18px 22px;background:var(--panel);border-bottom:1px solid var(--line)}}h1{{font-size:22px;margin:0 0 8px}}.stats,.controls{{display:flex;gap:8px;flex-wrap:wrap;align-items:center}}.stat{{color:var(--muted)}}.controls{{padding:12px 22px;border-bottom:1px solid var(--line);background:var(--panel)}}label{{font-weight:500}}input,select{{font:inherit;color:inherit;background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:7px 9px}}input{{min-width:220px}}main{{display:grid;grid-template-columns:minmax(0,1fr) 340px;min-height:680px}}#canvas{{overflow:auto;padding:18px}}svg{{display:block;background:var(--panel);border:1px solid var(--line);border-radius:8px}}aside{{background:var(--panel);border-left:1px solid var(--line);padding:18px;overflow-wrap:anywhere}}aside h2{{font-size:17px;margin:0 0 10px}}.edge{{fill:none;stroke:var(--line);stroke-width:1.5}}.edge.support{{stroke:var(--support)}}.edge.challenge{{stroke:var(--challenge);stroke-dasharray:5 4}}.edge.verification{{stroke:var(--verify)}}.edge.evolution{{stroke:var(--evolve)}}.node rect{{fill:var(--panel);stroke:var(--line);stroke-width:1.5;rx:8}}.node.verified rect{{stroke:var(--support)}}.node.replayable rect{{stroke:var(--evolve)}}.node.contested rect{{stroke:var(--challenge);stroke-width:2}}.node.recorded rect{{stroke:var(--warn)}}.node text{{fill:var(--ink);font-size:12px;pointer-events:none}}.node .muted{{fill:var(--muted)}}.node:focus{{outline:none}}.node:focus rect{{stroke-width:3}}table{{width:100%;border-collapse:collapse}}td{{padding:7px 0;border-bottom:1px solid var(--line);vertical-align:top}}td:first-child{{width:96px;color:var(--muted)}}code{{font:12px ui-monospace,SFMono-Regular,Menlo,monospace}}ul{{padding-left:20px}}.pass{{color:var(--support)}}.fail{{color:var(--challenge)}}.legend{{margin-top:10px;color:var(--muted)}}.legend span{{white-space:nowrap;margin-right:12px}}.dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}}.empty{{padding:28px;color:var(--muted)}}@media(max-width:900px){{main{{grid-template-columns:1fr}}aside{{border-left:0;border-top:1px solid var(--line)}}}}
</style></head><body><header><h1>{title_text}</h1><div class="stats" id="stats"></div><div class="legend"><span><i class="dot" style="background:var(--support)"></i>support/verified</span><span><i class="dot" style="background:var(--challenge)"></i>refute/contested</span><span><i class="dot" style="background:var(--verify)"></i>review/reproduce</span><span><i class="dot" style="background:var(--evolve)"></i>evolution</span></div></header>
<section class="controls" aria-label="Graph filters"><label for="search">Search</label><input id="search" type="search" placeholder="statement, ID, or kind"><label for="kind">Kind</label><select id="kind"><option value="">All kinds</option></select><label for="proof">Verification</label><select id="proof"><option value="">All levels</option><option>verified</option><option>replayable</option><option>traceable</option><option>recorded</option><option>contested</option></select></section>
<main><section id="canvas" aria-label="Scientific evidence graph"></section><aside><h2>Selected scientific object</h2><div id="details">Select a node to inspect its evidence checks.</div></aside></main>
<script id="dag-data" type="application/json">{payload}</script><script>
const graph=JSON.parse(document.getElementById('dag-data').textContent),allNodes=graph.nodes||[],allEdges=graph.edges||[],byId=new Map(allNodes.map(n=>[n.id,n]));
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
const kind=document.getElementById('kind'),proof=document.getElementById('proof'),search=document.getElementById('search');
[...new Set(allNodes.map(n=>n.kind))].sort().forEach(v=>kind.insertAdjacentHTML('beforeend',`<option>${{esc(v)}}</option>`));
document.getElementById('stats').innerHTML=`<span class="stat">${{allNodes.length}} nodes</span><span class="stat">${{allEdges.length}} relations</span><span class="stat">DAG ${{graph.integrity.is_dag?'valid':'blocked'}}</span><span class="stat">closure ${{esc(graph.scientific_closure.status)}}</span><span class="stat"><code>${{esc(graph.commit.slice(0,12))}}</code></span>`;
function detail(n){{const checks=(n.proof.checks||[]).map(c=>`<li class="${{c.passed?'pass':'fail'}}">${{c.passed?'✓':'✗'}} ${{esc(c.label)}} <small>(${{esc(c.layer)}})</small></li>`).join('');document.getElementById('details').innerHTML=`<table><tr><td>Summary</td><td>${{esc(n.summary)}}</td></tr><tr><td>ID</td><td><code>${{esc(n.id)}}</code></td></tr><tr><td>Type</td><td>${{esc(n.kind)}} / ${{esc(n.state)}}</td></tr><tr><td>Proof</td><td><strong>${{esc(n.proof.level)}}</strong></td></tr><tr><td>Hash</td><td><code>${{esc(n.content_hash||'-')}}</code></td></tr><tr><td>Actor</td><td>${{esc((n.actor||{{}}).actor_id||'-')}}</td></tr></table><h3>Verification checks</h3><ul>${{checks||'<li>No specialized checks.</li>'}}</ul>`}}
function render(){{const q=search.value.trim().toLowerCase();const nodes=allNodes.filter(n=>(!kind.value||n.kind===kind.value)&&(!proof.value||n.proof.level===proof.value)&&(!q||`${{n.id}} ${{n.kind}} ${{n.summary}}`.toLowerCase().includes(q)));const ids=new Set(nodes.map(n=>n.id)),edges=allEdges.filter(e=>ids.has(e.source)&&ids.has(e.target));const buckets=new Map();nodes.forEach(n=>{{if(!buckets.has(n.phase))buckets.set(n.phase,[]);buckets.get(n.phase).push(n)}});[...buckets.values()].forEach(v=>v.sort((a,b)=>a.id.localeCompare(b.id)));const phases=[...buckets.keys()].sort((a,b)=>a-b),boxW=220,boxH=82,gapX=92,gapY=28,pad=34,maxRows=Math.max(1,...[...buckets.values()].map(v=>v.length)),width=Math.max(720,pad*2+phases.length*boxW+Math.max(0,phases.length-1)*gapX),height=Math.max(220,pad*2+maxRows*boxH+Math.max(0,maxRows-1)*gapY),pos=new Map();phases.forEach((p,col)=>buckets.get(p).forEach((n,row)=>pos.set(n.id,{{x:pad+col*(boxW+gapX),y:pad+row*(boxH+gapY)}})));const edgeSvg=edges.map(e=>{{const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)return'';const x1=a.x+boxW,y1=a.y+boxH/2,x2=b.x,y2=b.y+boxH/2,m=x1+(x2-x1)/2;return`<path class="edge ${{esc(e.category)}}" d="M${{x1}} ${{y1}} C${{m}} ${{y1}},${{m}} ${{y2}},${{x2}} ${{y2}}" marker-end="url(#arrow)"/>`}}).join('');const nodeSvg=nodes.map(n=>{{const p=pos.get(n.id),summary=n.summary.length>29?n.summary.slice(0,29)+'…':n.summary;return`<g class="node ${{esc(n.proof.level)}}" data-id="${{esc(n.id)}}" tabindex="0" role="button" aria-label="${{esc(n.kind+' '+n.summary)}}" transform="translate(${{p.x}},${{p.y}})"><rect width="${{boxW}}" height="${{boxH}}"/><text x="12" y="20"><tspan font-weight="600">${{esc(n.kind)}}</tspan></text><text class="muted" x="12" y="40">${{esc(summary)}}</text><text class="muted" x="12" y="60">${{esc(n.state)}} · ${{esc(n.proof.level)}}</text><text class="muted" x="12" y="76">${{esc(n.id.slice(0,25))}}</text></g>`}}).join('');const canvas=document.getElementById('canvas');canvas.innerHTML=nodes.length?`<svg width="${{width}}" height="${{height}}" viewBox="0 0 ${{width}} ${{height}}"><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>${{edgeSvg}}${{nodeSvg}}</svg>`:'<div class="empty">No nodes match these filters.</div>';canvas.querySelectorAll('.node').forEach(el=>{{const show=()=>detail(byId.get(el.dataset.id));el.addEventListener('click',show);el.addEventListener('keydown',e=>{{if(e.key==='Enter'||e.key===' '){{e.preventDefault();show()}}}})}});if(nodes[0])detail(nodes[0])}}
[search,kind,proof].forEach(el=>el.addEventListener(el===search?'input':'change',render));render();
</script></body></html>"""


def export_research_dag(
    repo: str | Path,
    destination: str | Path,
    *,
    ref: str = "HEAD",
    ara_roots: Sequence[str | Path] = (),
    disclose_summaries: bool = True,
) -> dict[str, Any]:
    graph = build_research_dag(
        repo,
        ref=ref,
        ara_roots=ara_roots,
        disclose_summaries=disclose_summaries,
    )
    output = Path(destination).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "research-dag.json"
    html_path = output / "research-dag.html"
    atomic_write_json(json_path, graph)
    atomic_write_text(html_path, render_research_dag_html(graph))
    return {
        "graph": graph,
        "json": json_path.as_posix(),
        "html": html_path.as_posix(),
    }


__all__ = [
    "RESEARCH_DAG_SCHEMA",
    "build_research_dag",
    "export_research_dag",
    "render_research_dag_html",
]
