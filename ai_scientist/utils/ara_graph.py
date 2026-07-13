"""Utilities for ARA exploration-graph DAG checks and visualization."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ai_scientist.protocol.graph import (
    analyze_exploration_graph,
    graph_with_dag_metadata,
)


def render_exploration_graph_html(
    graph: dict[str, Any],
    *,
    title: str = "XScientist Exploration DAG",
) -> str:
    """Render a self-contained SVG/HTML view of an exploration DAG."""

    graph_payload = graph_with_dag_metadata(graph)
    data_json = json.dumps(graph_payload, ensure_ascii=False, default=str).replace("</", "<\\/")
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
      overflow: auto;
      padding: 20px;
    }}
    svg {{
      min-width: 100%;
      background: #fff;
      border: 1px solid #d9dee8;
      border-radius: 8px;
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
    .node.buggy rect {{ stroke: var(--bad); }}
    .node text {{ font-size: 12px; fill: var(--ink); }}
    .node .muted {{ fill: var(--muted); }}
    .edge {{ stroke: var(--line); stroke-width: 1.4; fill: none; marker-end: url(#arrow); }}
    .empty {{
      padding: 24px;
      color: var(--muted);
      background: #fff;
      border: 1px solid #d9dee8;
      border-radius: 8px;
    }}
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
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid #d9dee8; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title_text}</h1>
    <div class="summary" id="summary"></div>
  </header>
  <main>
    <section id="canvas" aria-label="Exploration graph"></section>
    <aside>
      <h2 style="margin:0 0 12px;font-size:18px;">Selected node</h2>
      <div id="details">Select a node in the graph.</div>
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
    const explicitEdges = edges
      .filter(e => byId.has(String(e.parent)) && byId.has(String(e.child)))
      .map(e => [String(e.parent), String(e.child)]);
    const seen = new Set(explicitEdges.map(e => e.join('\\u0000')));
    for (const n of nodes) {{
      if (n.parent_id && byId.has(String(n.parent_id))) {{
        const key = String(n.parent_id) + '\\u0000' + String(n.id);
        if (!seen.has(key)) {{
          explicitEdges.push([String(n.parent_id), String(n.id)]);
          seen.add(key);
        }}
      }}
      for (const child of Array.isArray(n.children) ? n.children : []) {{
        if (byId.has(String(child))) {{
          const key = String(n.id) + '\\u0000' + String(child);
          if (!seen.has(key)) {{
            explicitEdges.push([String(n.id), String(child)]);
            seen.add(key);
          }}
        }}
      }}
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
    if (!nodes.length) {{
      canvas.innerHTML = '<div class="empty">No exploration nodes recorded.</div>';
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
      const esc = s => String(s ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
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
        const cls = ['node', n.is_seed_node ? 'seed' : '', n.is_buggy ? 'buggy' : ''].join(' ');
        const shortId = id.length > 24 ? id.slice(0, 24) + '...' : id;
        const stage = n.stage ? String(n.stage) : 'stage=-';
        return `<g class="${{cls}}" data-id="${{esc(id)}}" tabindex="0" role="button" aria-label="node ${{esc(id)}}" transform="translate(${{p.x}},${{p.y}})">
          <rect width="${{boxW}}" height="${{boxH}}"></rect>
          <text x="12" y="22"><tspan font-weight="700">${{esc(shortId)}}</tspan></text>
          <text class="muted" x="12" y="43">${{esc(stage)}}</text>
          <text class="muted" x="12" y="62">${{esc(metricText(n))}}</text>
          <text class="muted" x="12" y="78">${{n.is_buggy ? 'buggy' : 'ok'}}${{n.is_seed_node ? ' / seed' : ''}}</text>
        </g>`;
      }}).join('');
      canvas.innerHTML = `<svg viewBox="0 0 ${{width}} ${{height}}" width="${{width}}" height="${{height}}">
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
        const rows = [
          ['id', `<code>${{esc(n.id)}}</code>`],
          ['hash', `<code>${{esc(n.content_hash || '-')}}</code>`],
          ['stage', esc(n.stage || '-')],
          ['step', esc(n.step ?? '-')],
          ['parent', `<code>${{esc(n.parent_id || '-')}}</code>`],
          ['children', esc((n.children || []).join(', ') || '-')],
          ['metric', esc(JSON.stringify(n.metric ?? '-'))],
          ['buggy', esc(n.is_buggy ?? '-')],
          ['seed', esc(n.is_seed_node ?? '-')],
          ['artifacts', `<code>${{esc(n.artifacts_dir || '-')}}</code>`],
          ['plan', esc(n.plan_excerpt || '-')]
        ];
        details.innerHTML = `<table>${{rows.map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td></tr>`).join('')}}</table>`;
      }};
      for (const el of canvas.querySelectorAll('.node')) {{
        const node = byId.get(el.dataset.id);
        el.addEventListener('click', () => show(node));
        el.addEventListener('keydown', ev => {{ if (ev.key === 'Enter' || ev.key === ' ') show(node); }});
      }}
      if (nodes[0]) show(nodes[0]);
    }}
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
    html_path = ara_path / html_name
    summary_path = ara_path / summary_name
    html_path.write_text(render_exploration_graph_html(graph_with_meta), encoding="utf-8")
    summary_path.write_text(
        json.dumps(graph_with_meta["dag"], indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return {
        "html": html_name,
        "summary": summary_name,
    }
