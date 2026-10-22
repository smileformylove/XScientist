"""Utilities for ARA exploration-graph DAG checks and visualization."""

from __future__ import annotations

import html
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

from ai_scientist.protocol.graph import (
    analyze_exploration_graph,
    graph_with_dag_metadata,
)

_PRESENTATION_ATTRIBUTE_KEYS = (
    "status",
    "decision",
    "stage",
    "step",
    "intent",
    "metric",
)
_MAX_PRESENTATION_EVENTS = 512
_MAX_PRESENTATION_TEXT = 240


def _presentation_text(value: Any) -> str:
    """Keep self-contained HTML payloads bounded without changing ledger data."""

    text = str(value or "")
    if len(text) <= _MAX_PRESENTATION_TEXT:
        return text
    return text[: _MAX_PRESENTATION_TEXT - 3] + "..."


def _presentation_events(events: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Return a bounded, presentation-only projection of admitted events.

    Event ledgers can contain arbitrary producer attributes and evidence
    hashes.  The graph page is a convenience view, not a disclosure boundary,
    so only stable identity fields and a small status allow-list are embedded
    in its self-contained HTML payload.  The append-only ledger remains the
    source of truth for complete event records.
    """

    projected: list[dict[str, Any]] = []
    source_events = list(events or [])
    for event in source_events[-_MAX_PRESENTATION_EVENTS:]:
        if not isinstance(event, dict):
            continue
        event_type = _presentation_text(event.get("event_type")).strip()
        event_id = _presentation_text(event.get("event_id")).strip()
        if not event_type or not event_id:
            continue
        subject = event.get("subject")
        if not isinstance(subject, dict):
            subject = {}
        subject_type = _presentation_text(subject.get("type")).strip()
        subject_id = _presentation_text(subject.get("id")).strip()
        attributes = event.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        safe_attributes: dict[str, Any] = {}
        for key in _PRESENTATION_ATTRIBUTE_KEYS:
            value = attributes.get(key)
            if isinstance(value, str):
                safe_attributes[key] = _presentation_text(value)
            elif value is None or isinstance(value, (int, float, bool)):
                if value is not None and (
                    not isinstance(value, float) or math.isfinite(value)
                ):
                    safe_attributes[key] = value
            elif key == "metric" and isinstance(value, dict):
                metric = {}
                if isinstance(value.get("name"), str):
                    metric["name"] = _presentation_text(value["name"])
                number = value.get("value")
                if type(number) in (int, float) and (
                    not isinstance(number, float) or math.isfinite(number)
                ):
                    metric["value"] = number
                if isinstance(value.get("maximize"), bool):
                    metric["maximize"] = value["maximize"]
                if metric:
                    safe_attributes[key] = metric
        projected.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "timestamp": _presentation_text(event.get("timestamp")),
                "actor": _presentation_text(event.get("actor")),
                "subject": {"type": subject_type, "id": subject_id},
                "attributes": safe_attributes,
                "relation_count": len(
                    [
                        item
                        for item in (
                            event["relations"]
                            if isinstance(event.get("relations"), list)
                            else []
                        )
                        if isinstance(item, dict)
                    ]
                ),
                "object_ref_count": len(
                    [
                        item
                        for item in (
                            event["object_refs"]
                            if isinstance(event.get("object_refs"), list)
                            else []
                        )
                        if isinstance(item, str)
                    ]
                ),
            }
        )
    return projected


def render_exploration_graph_html(
    graph: dict[str, Any],
    *,
    title: str = "XScientist Exploration DAG",
    events: list[dict[str, Any]] | None = None,
    latent_trajectory: dict[str, Any] | None = None,
    event_count: int | None = None,
) -> str:
    """Render a self-contained SVG/HTML view of an exploration DAG."""

    graph_payload = graph_with_dag_metadata(graph)
    # This is intentionally a separate presentation projection.  It must not
    # be merged into the canonical graph or its DAG diagnostics.
    event_source = list(
        events if events is not None else (graph.get("presentation_events") or [])
    )
    graph_payload["presentation_events"] = _presentation_events(event_source)
    total_event_count = max(len(event_source), event_count or 0)
    graph_payload["presentation_event_count"] = total_event_count
    graph_payload["presentation_events_truncated"] = (
        total_event_count > _MAX_PRESENTATION_EVENTS
    )
    if latent_trajectory is None and total_event_count == len(event_source):
        try:
            from ai_scientist.utils.latent_trajectory import (
                build_latent_trajectory_projection,
            )

            latent_trajectory = build_latent_trajectory_projection(event_source)
        except Exception:  # pragma: no cover - graph rendering remains usable
            latent_trajectory = None
    # Keep this projection separate from canonical DAG metadata.  It is an
    # explicitly declared intermediate-state view and never changes authority.
    from ai_scientist.utils.latent_trajectory import latent_trajectory_issues

    graph_payload["presentation_latent_trajectory"] = (
        latent_trajectory
        if latent_trajectory is not None
        and not latent_trajectory_issues(latent_trajectory)
        and latent_trajectory["source"]["event_count"] == total_event_count
        else None
    )
    data_json = json.dumps(graph_payload, ensure_ascii=False, default=str).replace(
        "</", "<\\/"
    )
    title_text = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title_text}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --ink: #18202f;
      --muted: #667085;
      --line: #c7cfdd;
      --ok: #16794c;
      --warn: #b65f00;
      --bad: #b42318;
      --seed: #3157a4;
      --node: #ffffff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 20px 24px 12px;
      border-bottom: 1px solid #d9dee8;
      background: #ffffff;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .summary {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .pill {{
      border: 1px solid #d6dbe6;
      border-radius: 999px;
      padding: 4px 9px;
      background: #fff;
    }}
    .pill.ok {{ color: var(--ok); border-color: #98d6b5; }}
    .pill.bad {{ color: var(--bad); border-color: #f0a8a2; }}
    main {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 340px;
      min-height: calc(100vh - 88px);
    }}
    #canvas {{
      min-width: 0;
      padding: 16px;
    }}
    .graph-toolbar {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 6px;
      margin-bottom: 8px;
    }}
    .graph-toolbar button {{
      min-width: 36px;
      height: 34px;
      border: 1px solid #c7cfdd;
      border-radius: 6px;
      padding: 0 10px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    .graph-toolbar button:hover {{ background: #f1f4f9; }}
    .graph-toolbar button:disabled {{
      color: #98a2b3;
      cursor: default;
      background: #f7f8fb;
    }}
    .graph-toolbar button[aria-pressed="true"] {{
      color: var(--seed);
      border-color: var(--seed);
      background: #eef4ff;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px 12px;
      margin: 0 0 8px;
      color: var(--muted);
      font-size: 12px;
    }}
    .legend-title {{ font-weight: 600; color: var(--ink); }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
    .legend-swatch {{
      display: inline-block;
      width: 22px;
      height: 12px;
      border: 1.5px solid #98a2b3;
      border-radius: 4px;
      background: var(--node);
    }}
    .legend-swatch.seed {{ border-color: var(--seed); }}
    .legend-swatch.rejected {{ border-color: var(--warn); border-style: dashed; }}
    .legend-swatch.buggy {{ border-color: var(--bad); }}
    .legend-swatch.edge {{ width: 22px; height: 0; border: 0; border-top: 2px solid var(--line); border-radius: 0; }}
    .legend-note {{ flex-basis: 100%; color: var(--muted); }}
    .zoom-level {{
      width: 58px;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      text-align: center;
    }}
    #graph-viewport {{
      height: clamp(360px, calc(100vh - 190px), 760px);
      overflow: auto;
      overscroll-behavior: contain;
      border: 1px solid #d9dee8;
      border-radius: 8px;
      background: #fff;
      cursor: grab;
    }}
    #graph-viewport.dragging {{
      cursor: grabbing;
      user-select: none;
    }}
    #graph-viewport svg {{
      display: block;
      min-width: 0;
      max-width: none;
      margin: 0 auto;
    }}
    #canvas.is-empty .graph-toolbar {{ display: none; }}
    #canvas.is-empty #graph-viewport {{
      height: auto;
      overflow: visible;
      border: 0;
      cursor: default;
    }}
    aside {{
      border-left: 1px solid #d9dee8;
      background: #fff;
      padding: 18px;
      overflow: auto;
    }}
    .node rect {{
      fill: var(--node);
      stroke: #98a2b3;
      stroke-width: 1.25;
      rx: 8;
    }}
    .node.seed rect {{ stroke: var(--seed); }}
    .node.scientific-rejected rect {{
      stroke: var(--warn);
      stroke-dasharray: 4 3;
    }}
    .node.buggy rect {{
      stroke: var(--bad);
      stroke-dasharray: none;
    }}
    .node.selected rect {{
      stroke: var(--seed);
      stroke-width: 2.5;
    }}
    .node text {{ font-size: 12px; fill: var(--ink); }}
    .node {{ cursor: pointer; }}
    .node .muted {{ fill: var(--muted); }}
    .edge {{ stroke: var(--line); stroke-width: 1.4; fill: none; marker-end: url(#arrow); }}
    .empty {{
      padding: 24px;
      color: var(--muted);
      background: #fff;
      border: 1px solid #d9dee8;
      border-radius: 8px;
    }}
    .state-panel {{
      margin-top: 22px;
      padding-top: 18px;
      border-top: 1px solid #d9dee8;
    }}
    .state-panel h2 {{ margin: 0 0 8px; font-size: 18px; }}
    .state-panel h3 {{ margin: 16px 0 6px; font-size: 13px; }}
    .state-note {{ margin: 0 0 10px; color: var(--muted); font-size: 12px; line-height: 1.4; }}
    .state-filter {{
      width: 100%;
      min-height: 34px;
      border: 1px solid #c7cfdd;
      border-radius: 6px;
      padding: 6px 8px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }}
    .state-summary {{ margin-top: 6px; color: var(--muted); font-size: 12px; }}
    .state-list {{
      display: grid;
      gap: 6px;
      margin: 0;
      padding: 0;
      list-style: none;
      font-size: 12px;
      max-height: 240px;
      overflow-y: auto;
      overscroll-behavior: contain;
    }}
    .state-item {{
      padding: 7px 8px;
      border: 1px solid #e4e8ef;
      border-radius: 6px;
      background: #fbfcfe;
      overflow-wrap: anywhere;
    }}
    .state-item .muted {{ color: var(--muted); }}
    .node.filtered-out {{ opacity: 0.16; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    td {{
      border-bottom: 1px solid #eef1f5;
      padding: 8px 0;
      vertical-align: top;
    }}
    td:first-child {{ color: var(--muted); width: 108px; }}
    td:last-child,
    code {{
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    button:focus-visible,
    #graph-viewport:focus-visible,
    .node:focus-visible,
    .state-filter:focus-visible {{
      outline: 3px solid #84adff;
      outline-offset: 2px;
    }}
    .state-list:focus-visible {{ outline: 2px solid #84adff; outline-offset: 2px; }}
    .sr-only {{
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid #d9dee8; }}
      #canvas {{ padding: 12px; }}
      #graph-viewport {{ height: clamp(320px, 56vh, 560px); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title_text}</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section id="canvas" aria-label="Exploration graph">
      <div class="graph-toolbar" role="toolbar" aria-label="Graph view controls">
        <button id="zoom-out" type="button" aria-label="Zoom out" title="Zoom out">-</button>
        <output id="zoom-level" class="zoom-level" aria-live="polite">100%</output>
        <button id="zoom-in" type="button" aria-label="Zoom in" title="Zoom in">+</button>
        <button id="zoom-fit" type="button" aria-label="Fit graph to view" title="Fit graph to view" aria-pressed="false">Fit</button>
        <button id="zoom-reset" type="button" aria-label="Reset graph to 100 percent" title="Reset graph to 100 percent" aria-pressed="false">100%</button>
      </div>
      <div class="legend" data-presentation-only="true" aria-label="Rendering legend">
        <span class="legend-title">Rendering legend</span>
        <span class="legend-item"><span class="legend-swatch edge"></span>relation</span>
        <span class="legend-item"><span class="legend-swatch seed"></span>seed node</span>
        <span class="legend-item"><span class="legend-swatch rejected"></span>scientific outcome rejected</span>
        <span class="legend-item"><span class="legend-swatch buggy"></span>execution failed</span>
        <span class="legend-note">Presentation only; labels reflect fields in the bound graph payload and do not add scientific evidence.</span>
      </div>
      <div id="graph-view-help" class="sr-only">Use the arrow keys or scroll to pan. Use plus and minus to zoom, F to fit, and zero to reset.</div>
      <div id="graph-viewport" tabindex="0" aria-label="Pannable exploration graph" aria-describedby="graph-view-help"></div>
    </section>
    <aside>
      <h2 style="margin:0 0 12px;font-size:18px;">Selected node</h2>
      <div id="details">Select a node in the graph.</div>
      <section class="state-panel" data-presentation-only="true" aria-label="Intermediate state">
        <h2>Intermediate state</h2>
        <p class="state-note">Presentation only; this ledger is not a scientific authority or ranking.</p>
        <label class="sr-only" for="state-filter">Filter nodes and state events</label>
        <input id="state-filter" class="state-filter" type="search" autocomplete="off"
          placeholder="Filter node or event state" aria-label="Filter nodes and state events">
        <div id="state-summary" class="state-summary" aria-live="polite"></div>
        <h3>Candidate frontier (DAG leaves)</h3>
        <ol id="frontier-list" class="state-list" tabindex="0" aria-label="Candidate frontier"></ol>
        <h3>Admitted events</h3>
        <ol id="event-list" class="state-list" tabindex="0" aria-label="Event ledger"></ol>
      </section>
      <section class="state-panel latent-panel" data-presentation-only="true" aria-label="Latent dynamics and belief trajectory">
        <h2>Latent dynamics / belief trajectory</h2>
        <p class="state-note">Explicit producer-declared fields only; unavailable means no latent trajectory was recorded.</p>
        <div id="latent-summary" class="state-summary" aria-live="polite"></div>
        <ol id="latent-list" class="state-list" tabindex="0" aria-label="Latent and belief steps"></ol>
      </section>
    </aside>
  </main>
  <script id="graph-data" type="application/json">{data_json}</script>
  <script>
    const graph = JSON.parse(document.getElementById('graph-data').textContent);
    const nodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    const edges = Array.isArray(graph.edges) ? graph.edges : [];
    const dag = graph.dag || {{}};
    const byId = new Map(nodes.map(n => [String(n.id), n]));
    const order = Array.isArray(dag.topological_order) && dag.topological_order.length
      ? dag.topological_order
      : nodes.map(n => String(n.id));
    const rank = new Map(order.map((id, i) => [id, i]));
    const explicitEdges = [];
    const seen = new Set();
    const addEdge = (parent, child) => {{
      parent = String(parent);
      child = String(child);
      const key = parent + '\\u0000' + child;
      if (byId.has(parent) && byId.has(child) && !seen.has(key)) {{
        explicitEdges.push([parent, child]);
        seen.add(key);
      }}
    }};
    for (const edge of edges) addEdge(edge.parent, edge.child);
    const useNodeMirrors = graph.topology_encoding !== 'edges';
    if (useNodeMirrors) {{
      for (const n of nodes) {{
        if (n.parent_id) addEdge(n.parent_id, n.id);
        for (const child of Array.isArray(n.children) ? n.children : []) {{
          addEdge(n.id, child);
        }}
      }}
    }}
    const relatives = new Map(nodes.map(n => [String(n.id), {{parents: [], children: []}}]));
    for (const [parent, child] of explicitEdges) {{
      relatives.get(parent).children.push(child);
      relatives.get(child).parents.push(parent);
    }}
    const incoming = new Map(nodes.map(n => [String(n.id), 0]));
    for (const [p, c] of explicitEdges) incoming.set(c, (incoming.get(c) || 0) + 1);
    const level = new Map();
    for (const id of order) {{
      if (!level.has(id)) level.set(id, 0);
      for (const [p, c] of explicitEdges.filter(e => e[0] === id)) {{
        level.set(c, Math.max(level.get(c) || 0, (level.get(id) || 0) + 1));
      }}
    }}
    const buckets = new Map();
    for (const n of nodes) {{
      const id = String(n.id);
      const l = level.get(id) || 0;
      if (!buckets.has(l)) buckets.set(l, []);
      buckets.get(l).push(n);
    }}
    for (const bucket of buckets.values()) {{
      bucket.sort((a, b) => (rank.get(String(a.id)) || 0) - (rank.get(String(b.id)) || 0));
    }}

    const summary = document.getElementById('summary');
    const pill = (text, cls='') => `<span class="pill ${{cls}}">${{text}}</span>`;
    summary.innerHTML = [
      pill(dag.is_dag ? 'DAG verified' : 'DAG issues', dag.is_dag ? 'ok' : 'bad'),
      pill(`${{nodes.length}} nodes`),
      pill(`${{explicitEdges.length}} edges`),
      pill(`${{(dag.root_ids || []).length}} roots`),
      pill(`${{(dag.leaf_ids || []).length}} leaves`),
      pill(`depth ${{dag.max_depth ?? '-'}}`)
    ].join('');

    const canvas = document.getElementById('canvas');
    const viewport = document.getElementById('graph-viewport');
    const presentationEvents = Array.isArray(graph.presentation_events)
      ? graph.presentation_events : [];
    const stateFilter = document.getElementById('state-filter');
    const stateSummary = document.getElementById('state-summary');
    const frontierList = document.getElementById('frontier-list');
    const eventList = document.getElementById('event-list');
    const latentTrajectory = graph.presentation_latent_trajectory;
    const latentSummary = document.getElementById('latent-summary');
    const latentList = document.getElementById('latent-list');
    const eventWindowNote = graph.presentation_events_truncated
      ? ' · latest 512 events shown' : '';
    const esc = s => String(s ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    const frontierIds = new Set((dag.leaf_ids || []).map(id => String(id)));
    const frontierNodes = nodes.filter(n => frontierIds.has(String(n.id)));
    const nodeSearchText = n => JSON.stringify({{
      id: n.id, stage: n.stage, plan: n.plan_excerpt, metric: n.metric,
      outcome: n.scientific_outcome, buggy: n.is_buggy
    }}).toLowerCase();
    const eventSearchText = event => JSON.stringify(event).toLowerCase();
    const renderStateLedger = () => {{
      const query = String(stateFilter?.value || '').trim().toLowerCase();
      const matchingFrontier = frontierNodes.filter(n => !query || nodeSearchText(n).includes(query));
      const matchingEvents = presentationEvents.filter(event => !query || eventSearchText(event).includes(query));
      frontierList.innerHTML = matchingFrontier.length
        ? matchingFrontier.map(n => {{
            const status = n.is_buggy === true ? 'execution failed'
              : (n.is_buggy === false ? 'execution ok' : 'execution unknown');
            return `<li class="state-item"><code>${{esc(n.id)}}</code>
              <span class="muted">${{esc(n.stage || 'stage=-')}} / ${{esc(status)}}</span></li>`;
          }}).join('')
        : '<li class="state-item muted">No matching DAG leaves.</li>';
      eventList.innerHTML = matchingEvents.length
        ? matchingEvents.map(event => {{
            const subject = event.subject || {{}};
            const attributes = event.attributes || {{}};
            const detail = Object.entries(attributes)
              .map(([key, value]) => `${{key}}=${{typeof value === 'object' ? JSON.stringify(value) : value}}`)
              .join(' · ');
            return `<li class="state-item"><strong>${{esc(event.event_type)}}</strong>
              <span class="muted">${{esc(subject.type || 'subject')}}:${{esc(subject.id || '-')}}
              / ${{esc(event.timestamp || '-')}}</span>
              ${{detail ? `<div class="muted">${{esc(detail)}}</div>` : ''}}</li>`;
          }}).join('')
        : '<li class="state-item muted">No matching admitted events.</li>';
      stateSummary.textContent = `${{matchingFrontier.length}} frontier leaves · ${{matchingEvents.length}} admitted events${{eventWindowNote}}`;
      for (const el of viewport.querySelectorAll('.node')) {{
        const node = byId.get(el.dataset.id);
        const visible = !query || (node && nodeSearchText(node).includes(query));
        el.classList.toggle('filtered-out', !visible);
        el.setAttribute('aria-hidden', String(!visible));
      }}
    }};
    const renderLatentTrajectory = () => {{
      if (!latentTrajectory || typeof latentTrajectory !== 'object') {{
        latentSummary.textContent = 'unavailable · projection failed closed';
        latentList.innerHTML = '<li class="state-item muted">Latent trajectory unavailable.</li>';
        return;
      }}
      const status = String(latentTrajectory.status || 'unavailable');
      const steps = Array.isArray(latentTrajectory.steps) ? latentTrajectory.steps : [];
      const source = latentTrajectory.source || {{}};
      const warnings = Array.isArray(latentTrajectory.warnings) ? latentTrajectory.warnings : [];
      latentSummary.textContent = `${{status}} · ${{steps.length}} observed steps · ${{source.event_count || 0}} ledger events${{warnings.length ? ' · ' + warnings.join(', ') : ''}}`;
      latentList.innerHTML = steps.length
        ? steps.map(step => {{
            const components = Array.isArray(step.components) ? step.components.join(', ') : '-';
            const subject = step.subject || {{}};
            const transition = step.transition || null;
            const edge = transition ? ` · ${{transition.from_state_id}} → ${{transition.to_state_id}}` : '';
            return `<li class="state-item"><strong>#${{esc(step.sequence)}} ${{esc(step.event_type)}}</strong>
              <span class="muted">${{esc(subject.type || 'subject')}}:${{esc(subject.id || '-')}} · ${{esc(components)}}${{esc(edge)}}</span></li>`;
          }}).join('')
        : '<li class="state-item muted">No explicit latent, belief, action, observation, or transition fields recorded.</li>';
    }};
    stateFilter.addEventListener('input', renderStateLedger);
    if (!nodes.length) {{
      canvas.classList.add('is-empty');
      viewport.innerHTML = '<div class="empty">No exploration nodes recorded.</div>';
    }} else {{
      const boxW = 220, boxH = 86, gapX = 96, gapY = 36, pad = 36;
      const maxLevel = Math.max(0, ...Array.from(buckets.keys()));
      const maxRows = Math.max(1, ...Array.from(buckets.values()).map(v => v.length));
      const width = pad * 2 + (maxLevel + 1) * boxW + maxLevel * gapX;
      const height = pad * 2 + maxRows * boxH + (maxRows - 1) * gapY;
      const pos = new Map();
      for (const [l, bucket] of buckets) {{
        bucket.forEach((n, row) => {{
          pos.set(String(n.id), {{
            x: pad + l * (boxW + gapX),
            y: pad + row * (boxH + gapY)
          }});
        }});
      }}
      const metricText = n => {{
        const m = n.metric;
        if (m && typeof m === 'object' && 'value' in m) return `metric=${{m.value}}`;
        if (m !== null && m !== undefined) return `metric=${{m}}`;
        return 'metric=-';
      }};
      const edgeSvg = explicitEdges.map(([p, c]) => {{
        const a = pos.get(p), b = pos.get(c);
        if (!a || !b) return '';
        const x1 = a.x + boxW, y1 = a.y + boxH / 2;
        const x2 = b.x, y2 = b.y + boxH / 2;
        const mid = x1 + Math.max(30, (x2 - x1) / 2);
        return `<path class="edge" d="M ${{x1}} ${{y1}} C ${{mid}} ${{y1}}, ${{mid}} ${{y2}}, ${{x2}} ${{y2}}" />`;
      }}).join('');
      const nodeSvg = nodes.map(n => {{
        const id = String(n.id);
        const p = pos.get(id);
        if (!p) return '';
        const scientificOutcome = n.scientific_outcome && typeof n.scientific_outcome === 'object'
          ? JSON.stringify(n.scientific_outcome)
          : String(n.scientific_outcome ?? '');
        const scientificRejected = scientificOutcome.toLowerCase().startsWith('rejected');
        const executionState = n.is_buggy === true
          ? 'execution failed'
          : (n.is_buggy === false ? 'execution ok' : 'execution unknown');
        const outcomeLabel = scientificOutcome.length > 28
          ? scientificOutcome.slice(0, 28) + '...'
          : scientificOutcome;
        const cls = [
          'node',
          n.is_seed_node ? 'seed' : '',
          n.is_buggy === true ? 'buggy' : '',
          scientificRejected ? 'scientific-rejected' : ''
        ].join(' ');
        const shortId = id.length > 24 ? id.slice(0, 24) + '...' : id;
        const stage = n.stage ? String(n.stage) : 'stage=-';
        return `<g class="${{cls}}" data-id="${{esc(id)}}" tabindex="0" role="button" aria-label="node ${{esc(id)}}" aria-pressed="false" transform="translate(${{p.x}},${{p.y}})">
          <rect width="${{boxW}}" height="${{boxH}}"></rect>
          <text x="12" y="22"><tspan font-weight="700">${{esc(shortId)}}</tspan></text>
          <text class="muted" x="12" y="43">${{esc(stage)}}</text>
          <text class="muted" x="12" y="62">${{esc(metricText(n))}}</text>
          <text class="muted" x="12" y="78">${{esc(outcomeLabel || executionState)}}${{n.is_seed_node ? ' / seed' : ''}}</text>
        </g>`;
      }}).join('');
      viewport.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="Exploration DAG with ${{nodes.length}} nodes">
        <defs>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#c7cfdd"></path>
          </marker>
        </defs>
        ${{edgeSvg}}
        ${{nodeSvg}}
      </svg>`;
      const details = document.getElementById('details');
      const show = n => {{
        const nodeRelatives = relatives.get(String(n.id)) || {{parents: [], children: []}};
        const rows = [
          ['id', `<code>${{esc(n.id)}}</code>`],
          ['hash', `<code>${{esc(n.content_hash || '-')}}</code>`],
          ['stage', esc(n.stage || '-')],
          ['step', esc(n.step ?? '-')],
          ['parent', `<code>${{esc(nodeRelatives.parents.join(', ') || '-')}}</code>`],
          ['children', esc(nodeRelatives.children.join(', ') || '-')],
          ['metric', esc(JSON.stringify(n.metric ?? '-'))],
          ['scientific outcome', esc(n.scientific_outcome && typeof n.scientific_outcome === 'object' ? JSON.stringify(n.scientific_outcome) : (n.scientific_outcome || '-'))],
          ['buggy', esc(n.is_buggy ?? '-')],
          ['seed', esc(n.is_seed_node ?? '-')],
          ['artifacts', `<code>${{esc(n.artifacts_dir || '-')}}</code>`],
          ['plan', esc(n.plan_excerpt || '-')]
        ];
        details.innerHTML = `<table>${{rows.map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td></tr>`).join('')}}</table>`;
        for (const el of viewport.querySelectorAll('.node')) {{
          const selected = el.dataset.id === String(n.id);
          el.classList.toggle('selected', selected);
          el.setAttribute('aria-pressed', String(selected));
        }}
      }};
      for (const el of viewport.querySelectorAll('.node')) {{
        const node = byId.get(el.dataset.id);
        el.addEventListener('click', () => show(node));
        el.addEventListener('keydown', ev => {{
          if (ev.key === 'Enter' || ev.key === ' ') {{
            ev.preventDefault();
            show(node);
          }}
        }});
      }}

      const svg = viewport.querySelector('svg');
      const zoomOut = document.getElementById('zoom-out');
      const zoomIn = document.getElementById('zoom-in');
      const zoomFit = document.getElementById('zoom-fit');
      const zoomReset = document.getElementById('zoom-reset');
      const zoomLevel = document.getElementById('zoom-level');
      const maxZoom = 3;
      let minZoom = 0.05;
      let zoom = 1;
      let fitMode = false;

      const updateZoomUi = () => {{
        zoomLevel.textContent = `${{Math.round(zoom * 100)}}%`;
        zoomOut.disabled = zoom <= minZoom + 0.001;
        zoomIn.disabled = zoom >= maxZoom - 0.001;
        zoomFit.setAttribute('aria-pressed', String(fitMode));
        zoomReset.setAttribute(
          'aria-pressed', String(!fitMode && Math.abs(zoom - 1) < 0.001)
        );
      }};
      const renderZoom = () => {{
        svg.style.width = `${{width * zoom}}px`;
        svg.style.height = `${{height * zoom}}px`;
        updateZoomUi();
      }};
      const setZoom = (nextZoom, anchorX, anchorY) => {{
        const x = anchorX ?? viewport.clientWidth / 2;
        const y = anchorY ?? viewport.clientHeight / 2;
        const oldScrollWidth = Math.max(1, viewport.scrollWidth);
        const oldScrollHeight = Math.max(1, viewport.scrollHeight);
        const xRatio = (viewport.scrollLeft + x) / oldScrollWidth;
        const yRatio = (viewport.scrollTop + y) / oldScrollHeight;
        zoom = Math.max(minZoom, Math.min(maxZoom, nextZoom));
        fitMode = false;
        renderZoom();
        viewport.scrollLeft = xRatio * viewport.scrollWidth - x;
        viewport.scrollTop = yRatio * viewport.scrollHeight - y;
      }};
      const fitGraph = () => {{
        const availableWidth = Math.max(1, viewport.clientWidth - 2);
        const availableHeight = Math.max(1, viewport.clientHeight - 2);
        const fitZoom = Math.min(availableWidth / width, availableHeight / height, 1);
        minZoom = fitZoom;
        zoom = fitZoom;
        fitMode = true;
        renderZoom();
        viewport.scrollLeft = 0;
        viewport.scrollTop = 0;
      }};
      const resetGraph = () => {{
        minZoom = Math.min(
          1,
          Math.min(
            Math.max(1, viewport.clientWidth - 2) / width,
            Math.max(1, viewport.clientHeight - 2) / height
          )
        );
        zoom = 1;
        fitMode = false;
        renderZoom();
        viewport.scrollLeft = 0;
        viewport.scrollTop = 0;
      }};

      zoomOut.addEventListener('click', () => setZoom(zoom / 1.25));
      zoomIn.addEventListener('click', () => setZoom(zoom * 1.25));
      zoomFit.addEventListener('click', fitGraph);
      zoomReset.addEventListener('click', resetGraph);
      viewport.addEventListener('wheel', ev => {{
        if (!ev.ctrlKey && !ev.metaKey) return;
        ev.preventDefault();
        const bounds = viewport.getBoundingClientRect();
        setZoom(
          zoom * (ev.deltaY < 0 ? 1.12 : 1 / 1.12),
          ev.clientX - bounds.left,
          ev.clientY - bounds.top
        );
      }}, {{passive: false}});
      viewport.addEventListener('keydown', ev => {{
        if (ev.target !== viewport) return;
        const panStep = 80;
        if (ev.key === '+' || ev.key === '=') {{
          ev.preventDefault();
          setZoom(zoom * 1.25);
        }} else if (ev.key === '-') {{
          ev.preventDefault();
          setZoom(zoom / 1.25);
        }} else if (ev.key === '0') {{
          ev.preventDefault();
          resetGraph();
        }} else if (ev.key === 'f' || ev.key === 'F') {{
          ev.preventDefault();
          fitGraph();
        }} else if (ev.key === 'ArrowLeft') {{
          ev.preventDefault();
          viewport.scrollLeft -= panStep;
        }} else if (ev.key === 'ArrowRight') {{
          ev.preventDefault();
          viewport.scrollLeft += panStep;
        }} else if (ev.key === 'ArrowUp') {{
          ev.preventDefault();
          viewport.scrollTop -= panStep;
        }} else if (ev.key === 'ArrowDown') {{
          ev.preventDefault();
          viewport.scrollTop += panStep;
        }}
      }});

      let drag = null;
      viewport.addEventListener('pointerdown', ev => {{
        if (ev.pointerType !== 'mouse' || ev.button !== 0) return;
        if (ev.target.closest && ev.target.closest('.node')) return;
        drag = {{
          x: ev.clientX,
          y: ev.clientY,
          left: viewport.scrollLeft,
          top: viewport.scrollTop
        }};
        viewport.classList.add('dragging');
        viewport.setPointerCapture(ev.pointerId);
      }});
      viewport.addEventListener('pointermove', ev => {{
        if (!drag) return;
        viewport.scrollLeft = drag.left - (ev.clientX - drag.x);
        viewport.scrollTop = drag.top - (ev.clientY - drag.y);
      }});
      const endDrag = ev => {{
        if (!drag) return;
        drag = null;
        viewport.classList.remove('dragging');
        if (viewport.hasPointerCapture(ev.pointerId)) {{
          viewport.releasePointerCapture(ev.pointerId);
        }}
      }};
      viewport.addEventListener('pointerup', endDrag);
      viewport.addEventListener('pointercancel', endDrag);

      let resizeFrame = 0;
      window.addEventListener('resize', () => {{
        if (!fitMode) return;
        window.cancelAnimationFrame(resizeFrame);
        resizeFrame = window.requestAnimationFrame(fitGraph);
      }});
      renderZoom();
      window.requestAnimationFrame(fitGraph);
      if (nodes[0]) show(nodes[0]);
    }}
    renderStateLedger();
    renderLatentTrajectory();
  </script>
</body>
</html>
"""


def write_exploration_graph_visualization(
    ara_root: str | Path,
    graph: dict[str, Any] | None = None,
    *,
    html_name: str = "exploration_graph.html",
    summary_name: str = "exploration_graph.summary.json",
) -> dict[str, str]:
    """Write ``exploration_graph.html`` and a machine-readable DAG summary."""

    ara_path = Path(ara_root)
    graph_payload = graph
    if graph_payload is None:
        graph_path = ara_path / "exploration_graph.json"
        graph_payload = json.loads(graph_path.read_text(encoding="utf-8"))
    if not isinstance(graph_payload, dict):
        graph_payload = {"nodes": [], "edges": []}

    graph_with_meta = graph_with_dag_metadata(graph_payload)
    # Read-only projection: an existing append-only event ledger enriches the
    # page without mutating the ARA or promoting events into DAG authority.
    presentation_events: deque[dict[str, Any]] = deque(maxlen=_MAX_PRESENTATION_EVENTS)
    event_count = 0

    def observed_events():
        nonlocal event_count
        from ai_scientist.utils.ara_events import iter_events

        for event in iter_events(ara_path):
            event_count += 1
            presentation_events.append(event)
            yield event

    try:
        from ai_scientist.utils.latent_trajectory import (
            build_latent_trajectory_projection,
        )

        presentation_latent_trajectory = build_latent_trajectory_projection(
            observed_events()
        )
    except Exception:  # pragma: no cover - visualization must remain usable
        presentation_latent_trajectory = None
    html_path = ara_path / html_name
    summary_path = ara_path / summary_name
    html_path.write_text(
        render_exploration_graph_html(
            graph_with_meta,
            events=list(presentation_events),
            latent_trajectory=presentation_latent_trajectory,
            event_count=event_count,
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(graph_with_meta["dag"], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "html": html_name,
        "summary": summary_name,
    }
