# ARA Protocol Specification

**Version:** `ara.v1`

An **Agent-Native Research Artifact (ARA)** is a directory that captures one
end-to-end research run in a form a downstream AI (or human) agent can *fork*,
*re-execute*, and *verify* without decoding the human-readable paper.

This document is the contract between producers (systems that write ARAs) and
consumers (systems that read them). If your implementation follows the rules
below, `ai_scientist.protocol.validate_ara(path)` will accept it.

Producers MUST NOT introduce new required fields without bumping
`PROTOCOL_VERSION`. Consumers MUST ignore unknown optional fields — the
protocol is designed to be additive.

## 1. Directory Layout

An ARA is a directory rooted at `manifest.json`:

```
<ara_root>/
    manifest.json                REQUIRED. Machine-readable entry point.
    exploration_graph.json       REQUIRED. Full node graph, including failures.
    README.md                    RECOMMENDED. Structured agent-facing entry.

    nodes/<node_id>/             OPTIONAL, but MUST match graph if present.
        code.py                  Exact code the node executed.
        term_out.log             Untruncated stdout/stderr.
        metrics.json             Per-node metric + analysis + `content_hash`.
        plots.json               Plot paths + VLM analyses.
        env.json                 Minimal env descriptor.
        run.sh                   POSIX runner. Executable.

    claims/<claim_id>.json       OPTIONAL. One per manuscript claim.
    claims/_index.json           OPTIONAL. Summary of claim resolution.

    verify/<node_id>_<ts>.json   OPTIONAL. Re-execution reports.
    verify/reexec_batch_*.json   OPTIONAL. Batched re-exec reports.

    env/                         OPTIONAL. Environment snapshot.
        bfts_config.yaml
        model_fingerprint.json
        requirements.freeze      (populated by `run_ara_fork.py freeze`)
```

The **only** files a validator requires are `manifest.json` and
`exploration_graph.json`. Everything else is best-effort: absent pieces MUST
be listed in `manifest.missing`.

## 2. Required File: `manifest.json`

The single source of truth. All other paths in the ARA are derivable from
here, so consumers should read `manifest.json` first.

Required top-level keys:

| Key | Type | Notes |
|---|---|---|
| `schema_version` | string | MUST equal `"ara.v1"` for this protocol version. |
| `protocol_kind` | string | RECOMMENDED. Consumers may assert `== "manifest"`. |
| `created_at` | string (ISO-8601) | UTC recommended. |
| `source_exp_dir` | string | Absolute path to the origin experiment directory. Purely informational — do not rely on it existing on the consumer's disk. |
| `idea` | object | Must contain `name`; other fields optional. |
| `counts` | object | Must contain `nodes`. `edges`, `buggy_nodes`, `journals`, `claims` recommended. |

Optional keys:

- `project_dir`: the parent directory this ARA landed under.
- `references`: producer-specific pointers to sub-artifacts.
- `missing`: array of strings describing intentionally-absent pieces.
- `provenance`: back-pointer to the parent ARA/node this run forked from. See §7.
- `counts_updated_at`: set when a later pass (e.g. claim resolution) mutated `counts`.

## 3. Required File: `exploration_graph.json`

The full tree-search graph. **Failed branches MUST be present** — omitting
them defeats the purpose of the protocol.

Required top-level keys: `schema_version`, `nodes`, `edges`, `counts`.

Each **node entry** in `nodes[]` must contain `id`. Recommended:
`content_hash`, `parent_id`, `children`, `is_buggy`, `metric`, `step`, `stage`.

Each **edge entry** in `edges[]` must contain `parent` and `child`.

`counts.nodes` MUST equal `len(nodes)`.

## 4. Content Addressing (`content_hash`)

Every node SHOULD carry a `content_hash` field of the form
`sha256:<64 hex chars>`. The hash is computed over a canonical JSON payload:

```json
{
  "code": "<node's code, stripped>",
  "code_len_bytes": 1234,
  "metric": {"name": ..., "value": ..., "maximize": ...},
  "extras": { ... optional producer-supplied fields ... }
}
```

For large code payloads (>256 KiB) the ``code`` field is truncated to the
byte cap and a ``code_truncated: true`` marker is added — the original
``code_len_bytes`` still stabilises the identity, so two different long
files hash differently.

Rules:

- Whitespace outside the code string is normalised (canonical JSON: sorted
  keys, no extra spaces).
- `metric.value` is coerced to `float` when possible.
- Only `name`, `value`, `maximize` are hashed from the metric object.
- `extras` is optional. Producers that want to bind additional stable inputs
  (dataset name, seed) SHOULD do so via `extras`; unstable inputs
  (timestamps, memory addresses) MUST NOT enter the payload.

Reference implementation: `ai_scientist.protocol.hash_node_payload`.

## 5. Per-Node Files (`nodes/<id>/`)

If a node's `code` is non-empty, its `nodes/<id>/` directory SHOULD contain:

- `code.py` — verbatim node code.
- `term_out.log` — untruncated combined stdout/stderr.
- `metrics.json` — see §5.1.
- `plots.json` — plot paths + analyses.
- `env.json` — minimal descriptor: at least `python_version`, `expected_cwd`,
  `code_file`.
- `run.sh` — POSIX runner that invokes the code. MUST be executable
  (`chmod 0755`).

### 5.1 `metrics.json`

```json
{
  "metric": {"name": "acc", "value": 0.82, "maximize": true},
  "analysis": "brief text",
  "is_buggy": false,
  "content_hash": "sha256:..."
}
```

The `content_hash` here MUST match the value in the corresponding
`exploration_graph.json` node entry.

## 6. Claims (`claims/<claim_id>.json`)

Each claim links one manuscript assertion to one exploration node:

```json
{
  "claim_id": "template_142_n1",
  "node_id": "n1",
  "tex_file": "template.tex",
  "line": 142,
  "context": "F1=0.82 on the held-out set...",
  "options": {"stage": "ablation"},
  "resolved": true,
  "node": { ... snapshot from exploration_graph.json ... }
}
```

Producers scan the manuscript source for `\claimref{<node_id>}` markers
(defined as a no-op LaTeX macro so the PDF stays clean). Unresolved claims
still MUST be written — with `resolved: false` and `node: null` — so
consumers can see the manuscript's *intent* even when the node is missing.

The optional `claims/_index.json` gives a batch summary and is treated as
non-schema-checked metadata.

## 7. Provenance (Fork Lineage)

When one ARA is derived from another (e.g. via `run_ara_fork.py fork`),
the child SHOULD record its origin:

```json
"provenance": {
  "parent_ara_root": "/path/to/parent/ara",
  "parent_node_id": "n47",
  "parent_content_hash": "sha256:..."
}
```

Content-hash provenance is preferable to path-based provenance: paths break
when directories move, hashes don't. A validator will accept either.

### 7.1 Fork-Continue Workflow

The end-to-end "one agent picks up where another left off" story:

```
# 1. Producer A finishes a run; ARA lands under <A_project>/ara/<...>/.
# 2. Anyone (human or another agent) picks a node and forks:
python run_ara_fork.py fork \
    --ara <A_project>/ara/<timestamp_idea>/ \
    --node-id <node_id> \
    --dest ./my_fork

# 3. Producer B seeds a fresh XScientist run from that fork:
python run_project.py \
    --project-dir <B_project> \
    --seed-from-ara ./my_fork \
    [other flags...]
```

Under the hood, `--seed-from-ara`:

1. Reads `./my_fork/node/code.py` + `./my_fork/fork.json`.
2. Stages a *seed manifest* (schema kind `"seed"`) under
   `<B_project>/.ara_seed/ara_seed.json`.
3. Sets `AI_SCIENTIST_ARA_SEED_PATH` so the BFTS ``_draft`` step returns the
   seeded ``Node`` instead of calling the LLM.
4. Passes the manifest's `provenance` block into producer B's `export_ara`
   call, so B's manifest points back at A's node via content hash.

The seed manifest itself is not part of the on-disk ARA — it lives under
`.ara_seed/` inside the child project. Consumers of an ARA only see
`provenance` in the manifest.

Alternate entry (no fork step, seed directly from an ARA node):

```
python run_project.py \
    --project-dir <B_project> \
    --seed-from-ara <A_project>/ara/<timestamp_idea>/ \
    --seed-node-id <node_id>
```

## 8. Verify Reports (`verify/*.json`)

Two shapes coexist under `verify/`:

- `ara.verify.v1` — one re-execution attempt against one node.
- `ara.reexec.batch.v1` — a set of verify reports produced in one pass.

Both formats are additive (`additionalProperties: true`) — future producers
can attach richer telemetry without breaking older consumers.

## 9. Conformance

Reference validator: `ai_scientist.protocol.validate_ara(path)`.

The validator returns a `ValidationReport` with `ok`, `errors`, `warnings`,
and `checked`. `strict=True` promotes warnings to errors — useful in CI.

To claim conformance a producer MUST pass:

- Presence of `manifest.json` and `exploration_graph.json`.
- Schema validity of both files.
- Every `nodes/<id>/metrics.json` (when present) schema-valid.
- Every `claims/*.json` (except `_index.json`) schema-valid.
- Every `verify/*.json` schema-valid.

Warnings (non-blocking):

- `schema_version` differs from `PROTOCOL_VERSION`.
- A node listed in `exploration_graph.json` has no directory on disk.

## 10. Non-Goals

The protocol deliberately does not specify:

- Bit-for-bit reproducibility. Producers with non-deterministic LLM calls
  will not achieve this; the protocol targets *scientific* reproducibility
  (same claim, same code, comparable metric).
- A shared identifier space across producers. Node ids are producer-local.
  Cross-producer identity is the job of `content_hash`.
- Any UI / rendering conventions. `README.md` is a courtesy for agents that
  don't want to parse `manifest.json` first — it is not authoritative.

## 11. Extension Points

Adding fields:
- **Add optional fields freely** — schemas are `additionalProperties: true`.
- **Add required fields only with a `PROTOCOL_VERSION` bump**.

Adding kinds:
- Extend `Kind` in `ai_scientist/protocol/constants.py`.
- Add the schema JSON under `ai_scientist/protocol/schemas/`.
- Update this document.

Adding producers:
- Copy this SPEC, implement the validator locally, and cross-check by round-
  tripping an example ARA through `ai_scientist.protocol.validate_ara`.
