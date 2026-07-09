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

    objects/sha256/<h>/<rest>    OPTIONAL. Content-addressable object store (§10).
    llm/calls.jsonl              OPTIONAL. One row per LLM invocation (§11).
    seed/ara_seed.json           OPTIONAL. Snapshot of consumed seed (§12).
    pipeline/<artifact>.json     OPTIONAL. Mirrored pipeline stage state (§13).
    manifest.lock                OPTIONAL. Base manifest hash + created_at (§14).
    manifest.history.jsonl       OPTIONAL. Append-only post-export edits (§14).
    history/<manifest_hash>.json OPTIONAL. Full snapshot per revision (§14).
    refs/<name>                  OPTIONAL. Local caller-writable bookmarks (§15).
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

### 7.0 Multi-Parent Provenance

A child ARA may inherit from *several* ancestors — code from A, env from B,
data hypothesis from C. Producers MAY populate a ``parents`` array to make
that explicit:

```json
"provenance": {
  "parent_ara_root": "/path/to/A",
  "parent_node_id": "n47",
  "parent_content_hash": "sha256:aaa...",
  "parents": [
    {"role": "code", "parent_ara_root": "/path/to/A", "parent_node_id": "n47",
     "parent_content_hash": "sha256:aaa..."},
    {"role": "env",  "parent_ara_root": "/path/to/B", "parent_node_id": "n03",
     "parent_content_hash": "sha256:bbb..."},
    {"role": "data", "parent_ara_root": "/path/to/C", "parent_node_id": "n11",
     "parent_content_hash": "sha256:ccc..."}
  ]
}
```

Rules:

- Consumers that understand ``parents`` MUST treat it as canonical.
- Consumers that don't SHOULD fall back to the top-level ``parent_*`` fields;
  ``ai_scientist.protocol.build_provenance`` echoes the ``role: "code"``
  entry into those slots by default, so single-parent readers still work.
- ``role`` is free-form. Common values: ``code`` / ``env`` / ``data`` /
  ``hypothesis``. Producers SHOULD stick to short lowercase tokens.

Reference helper: ``ai_scientist.protocol.build_provenance(...)``.

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

## 9. Metric Markers (Re-execution Contract)

Node code that wants its metric picked up by third-party verifiers MUST emit
a **metric marker line** on stdout of the form:

```
ARA_METRIC={"name": "<metric_name>", "value": <float>, "maximize": <bool>}
```

Rules:

- One JSON object per line, prefixed by the literal string ``ARA_METRIC=``.
- The last matching line wins (mid-run prints don't override the final one).
- ``value`` MUST be JSON-numeric. ``name`` and ``maximize`` are recommended
  but not strictly required.
- Additional fields are allowed — verifiers preserve them under
  ``fresh_metric.*`` but do NOT hash them.

Legacy fallback: a trailing line ``metric: 0.42`` (case-insensitive,
optional whitespace) is also accepted so historical scripts keep working.
New producers SHOULD emit ``ARA_METRIC=`` — it's unambiguous and pastes
into other logs cleanly.

Reference implementation: `ai_scientist.utils.ara_metric_parser`. Schema
file: `schemas/metric_marker.schema.json`.

## 10. Content-Addressable Object Store (`objects/`)

Any sizeable, immutable payload — LLM prompts and responses, mirrored
pipeline artifacts, large prompt fragments — is written to a two-level
sharded content-addressable store at:

```
<ara>/objects/sha256/<hex[:2]>/<hex[2:]>
```

Callers store only an `ObjectRef` in the referencing row, never the inline
payload:

```json
{"hash": "sha256:<64 hex>", "size": 12345, "gzip": true}
```

Payloads at or above 4 KiB are gzipped on disk; readers detect compression
from the gzip magic bytes (`1f 8b`), so no sidecar is needed. Writes are
atomic (`.tmp` → `os.replace`) and idempotent — writing the same bytes
twice yields the same hash and skips the second disk write.

The store is append-only. Once an object is written it is expected to live
for the lifetime of the ARA; garbage collection, if it ever ships, will be
a separate tool that walks manifests and prunes unreferenced blobs.

Reference implementation: `ai_scientist.protocol.objects.ObjectStore`.

## 11. LLM Call Log (`llm/calls.jsonl`)

Every model invocation is logged as one JSON line under
`<ara>/llm/calls.jsonl`. Required fields:

```json
{
  "schema_version": "ara.v1",
  "protocol_kind": "llm_call",
  "call_id": "<uuid>",
  "ts": "2026-07-10T12:00:00Z",
  "provider": "anthropic",
  "model": "anthropic/glm-5.1",
  "params": {"temperature": 0.7, "max_tokens": 4096},
  "messages_ref": {"hash": "sha256:...", "size": 12345, "gzip": true},
  "response_ref": {"hash": "sha256:...", "size":   234, "gzip": false},
  "tokens": {"input": 300, "output": 42},
  "latency_ms": 815
}
```

Message and response blobs live in `objects/` (§10) and are referenced by
`messages_ref` / `response_ref`. Rows MUST NOT inline prompt or response
text. Schema file: `schemas/llm_call.schema.json`.

### 11.1 Binding LLM calls into `content_hash`

Two additive node-level fields make LLM provenance first-class:

- **`llm_call_refs`** on `exploration_graph.nodes[]` — array of
  `messages_ref.hash` strings pointing back into `llm/calls.jsonl`. Purely
  informational; not required by the validator.
- **`content_hash_inputs`** on both `exploration_graph.nodes[]` and
  `nodes/<id>/metrics.json` — array declaring which categories fed the
  hash. Older ARAs omit this and are treated as `["code","metric"]`. Add
  `"llm_calls"` when LLM-call hashes were bound in via `extras`. Add
  `"seed"` when `Node.is_seed_node` is True; this ensures a seed-derived
  node with identical code+metric hashes differently from a regular
  exploration node so the semantic role stays part of the content address.

```json
{
  "content_hash": "sha256:...",
  "content_hash_inputs": ["code", "metric", "llm_calls"],
  "llm_call_refs": ["sha256:...", "sha256:..."]
}
```

A hash declared with `["code","metric","llm_calls"]` will not collide with
one declared `["code","metric"]` even for identical code — cross-producer
comparison stays sound. Consumers that don't understand these fields may
safely ignore them.

Schemas: `schemas/exploration_graph.schema.json`,
`schemas/node.schema.json`, `schemas/llm_call.schema.json`.

## 12. Seed Snapshot (`seed/`)

For runs launched via `--seed-from-ara`, the consumed seed manifest is
snapshotted at `<ara>/seed/ara_seed.json`. The snapshot is byte-identical
to the seed the producer read from `.ara_seed/` (§7.1) — it makes the
child ARA self-contained even if the source project directory is deleted
or relocated.

The manifest records the seed's content_hash under
`provenance.seed_hash`:

```json
"provenance": {
  "parent_ara_root": "/path/to/A",
  "parent_content_hash": "sha256:...",
  "seed_hash": "sha256:<64 hex — mirrors content_hash of ara_seed.json>"
}
```

Because `seed_hash` is content-addressed, consumers can dedup / cluster
forks that started from the same seed across producers that stored the
source ARA at different paths. The snapshot is also referenced from
`manifest.references.seed` using the shape described in §13.

## 13. Pipeline Artifact Mirror (`pipeline/`)

Producers that run through the pipeline_contracts stack mirror per-stage
artifacts under `<ara>/pipeline/` so consumers can inspect the pre-writeup
state that led to the final manuscript. Sixteen artifact kinds are
recognised (see the schema below for the full vocabulary); common paths:

```
<ara>/pipeline/
    review_state.json           critic_findings.json
    claim_evidence_graph.json   manuscript_state.json
    figure_spec.json            stage_standards.json
    process_alignment.json      research_program.json
    manuscript_candidate_pool.json
    self_evolution.json         research_plan.json
    idea_cards.json             pipeline_manifest.json
    repair_attempts.json        pareto_pool.json
    experiment_registry.json
```

Each mirrored artifact is registered as one entry in
`manifest.references.pipeline_artifacts[]`, following
`schemas/reference_manifest.schema.json`:

```json
{
  "kind": "review_state",
  "path": "pipeline/review_state.json",
  "schema_version": "review.v3",
  "content_hash": "sha256:...",
  "producer": "reviewer_agent",
  "generated_at": "2026-07-10T12:00:00Z",
  "size": 4123
}
```

Consumers that only care about a subset can filter by `kind`; those that
want cross-ARA diffs use `content_hash`.

## 14. Immutability Layer (`manifest.lock` + `manifest.history.jsonl`)

Once an ARA is exported, its base manifest is stamped and every subsequent
mutation is append-only.

- **`<ara>/manifest.lock`** — written once at export time. Records the
  `manifest_hash` of the ROOT (revision-0) manifest.json plus the
  algorithm name (`hasher`, default `"hash_manifest.v1"`) and
  `created_at`. Tampering with the base manifest is detectable by
  re-hashing and comparing.
- **`<ara>/manifest.history.jsonl`** — one JSON line per post-export
  mutation, produced by `append_manifest_revision(...)`. Fields:
  `revision` (1-indexed; base is 0), `ts`, `base_hash` (before),
  `new_hash` (after), `changed_fields` (informational), `reason`,
  `producer`.
- **`<ara>/history/<manifest_hash>.json`** — the full manifest snapshot
  after each revision, keyed by its `new_hash`. Audits can rewind to any
  prior state without replaying the diff chain.

```json
{
  "schema_version": "ara.v1",
  "protocol_kind": "manifest_revision",
  "revision": 3,
  "ts": "2026-07-10T13:00:00Z",
  "base_hash": "sha256:...",
  "new_hash":  "sha256:...",
  "changed_fields": ["counts.claims"],
  "reason": "claim_count updated after tex scan",
  "producer": "update_manifest_claim_count"
}
```

Producers MUST route every post-write manifest edit through
`append_manifest_revision` — direct writes to `manifest.json` break the
audit chain. Schemas: `schemas/manifest_lock.schema.json`,
`schemas/manifest_revision.schema.json`.

## 15. Local Refs (`refs/`)

`<ara>/refs/` is a git-style namespace of short human-readable names
pointing at content hashes. Each ref is a single text file whose sole
content is a `sha256:<hex>` line; nested paths are permitted:

```
<ara>/refs/candidates/best
<ara>/refs/ideas/paper_v3
<ara>/refs/pareto/frontier_2026Q2
```

Refs are **caller-writable** local bookmarks — CI, humans, or downstream
agents may create, update, or delete them at any time. They are NOT part
of `content_hash` and NOT covered by the immutability layer (§14). Ref
names are validated against directory traversal; targets must match
`^<algo>:[0-9a-f]+$`.

Reference implementation: `ai_scientist.utils.ara_refs` (`set_ref`,
`get_ref`, `delete_ref`, `list_refs`).

## 16. Signatures (`manifest.signatures[]`)

`manifest.signatures[]` is an optional array of detached signatures over
the base `manifest_hash` from `manifest.lock`. Each entry has `algo`,
`key_id`, `signature` (base64), plus optional `signed_at` and `signer`
(human-readable claim — verify against `key_id`, do not trust standalone):

```json
"signatures": [
  {"algo": "minisign", "key_id": "RWQ...", "signature": "...",
   "signed_at": "2026-07-10T14:00:00Z", "signer": "ci@xscientist.io"}
]
```

The `signatures` field is EXCLUDED from `hash_manifest()` — signing the
manifest does not invalidate its own subject. Consumers verify signatures
against their own key ring; the protocol treats each entry as opaque.

## 17. Conformance

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

## 18. Non-Goals

The protocol deliberately does not specify:

- Bit-for-bit reproducibility. Producers with non-deterministic LLM calls
  will not achieve this; the protocol targets *scientific* reproducibility
  (same claim, same code, comparable metric).
- A shared identifier space across producers. Node ids are producer-local.
  Cross-producer identity is the job of `content_hash`.
- Any UI / rendering conventions. `README.md` is a courtesy for agents that
  don't want to parse `manifest.json` first — it is not authoritative.

## 19. Extension Points

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
