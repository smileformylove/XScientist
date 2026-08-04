# ARA Storage and Lifecycle

ARA keeps scientific history immutable without requiring every payload to
remain duplicated in hot storage. The lifecycle model separates:

1. **Scientific metadata** — manifests, DAG edges, claim/evidence bindings,
   provenance, hashes, and verification state. These remain small and durable.
2. **Content-addressed objects** — prompts, responses, mirrored pipeline
   artifacts, verification output, and other immutable payloads.
3. **Derived views** — HTML graphs, summaries, counts, and indexes. These may
   be removed and regenerated.

The governing rule is **store completely, consume selectively**. Record volume
is not controlled by deleting scientific history; it is controlled by keeping
raw storage out of default model context.

## Canonical compact records

- New exploration graphs use `edges[]` as the sole topology source and set
  `topology_encoding: "edges"`. Readers still accept legacy `parent_id` and
  `children` fields.
- New claim files join the graph through `node_id`, `claim_hash`, and
  `evidence_refs`. The legacy embedded `node` snapshot is optional and is no
  longer written.
- Re-execution reports put stdout/stderr tails and compact verdict payloads in
  `objects/`; JSON reports retain ObjectRefs and comparison fields.
- Pipeline and seed CAS writes use `<project>/.ara-store/` as a shared backing
  store when exported by XScientist. Each ARA retains a local hard-linked view,
  so existing readers and portable audit bundles continue to work.

## Event ledger, semantic catalog, and ContextPack

`events/research_events.jsonl` admits only state-changing outcomes: completed
nodes, claim/evidence bindings, and verification results. Raw LLM/tool calls,
retry chatter, and terminal output remain stored in their native locations but
are not copied into this semantic reading order.

`catalog/semantic.sqlite` joins nodes, claims, graph relations, events, and
object links. The catalog is derived, carries a source fingerprint, and is
automatically rebuilt when stale:

```bash
xscientist ara catalog --ara <ara>
xscientist ara catalog --ara <ara> --rebuild
```

Consumers request an intent-specific view rather than opening the catalog or
the whole artifact:

```bash
xscientist ara context --ara <ara> --intent continue --node n17
xscientist ara context --ara <ara> --intent write --claim c12
xscientist ara context --ara <ara> --intent audit --claim c12
xscientist ara context --ara <ara> --intent reproduce --node n17 --receipt --json
```

The four views have different consumers and effects:

| Intent | Consumer | Operational use |
|---|---|---|
| `continue` | Experiment Agent | Inject current baseline, decisive evidence, failed attempts, and do-not-repeat items before node planning. |
| `write` | Writing Agent | Permit resolved evidence-backed claims and expose unsupported statements as hypotheses. |
| `audit` | Reviewer Agent | Present positive/negative evidence, unresolved claims, verification status, and omissions. |
| `reproduce` | Reproduce executor | Supply code, environment, run hook, expected outputs, and verification rules. |

The normal pipeline injects these packs automatically. New nodes record
`context_pack_refs`; claims record the writer packs; verify reports record the
reproduction pack. `context/receipts.jsonl` therefore answers which stored
information a consumer actually saw without duplicating that information.

Budgets trim optional context only. Target identity, evidence references,
execution dependencies, and verification rules are a hard closure and are
never removed to fit a prompt budget.

## Bundle profiles

`xscientist ara bundle` computes a file/object closure rather than assuming
every consumer needs the full directory:

| Profile | Guarantee |
|---|---|
| `index` | Manifest, graph, claims, and immutable history for inspection. |
| `fork` | Index plus selected executable node and environment/seed material. |
| `reproduce` | Fork material plus verification and reproduction-critical pipeline artifacts. |
| `audit` | Every non-GC file in the ARA; this is the default for compatibility. |

Every archive contains `bundle.manifest.json`, including the selected nodes,
claims, object references, missing references, and a `complete` flag. Fork and
reproduce bundles fail closed on missing selected objects unless the caller
passes `--allow-incomplete`.

```bash
xscientist ara bundle --ara <ara> --dest index.tar.gz --profile index
xscientist ara bundle --ara <ara> --dest fork.tar.gz --profile fork --node n17
xscientist ara bundle --ara <ara> --dest repro.tar.gz --profile reproduce --claim c12
```

## Storage inspection and pins

```bash
xscientist ara storage-report --ara <ara>
xscientist ara storage-report --ara <ara> --json

xscientist ara pin --ara <ara> --name release/paper-v1 --set sha256:<hash>
xscientist ara pin --ara <ara> --list
xscientist ara pin --ara <ara> --name release/paper-v1 --delete
```

Pins reuse the caller-local `refs/pins/*` namespace. A pinned object is a GC
root and cannot appear in a new collection plan.

If an ARA-local hard link has been removed while the project shared store is
still available, restore it with:

```bash
xscientist ara hydrate --ara <ara>
xscientist ara hydrate --ara <ara> --hash sha256:<hash>
```

The storage report distinguishes physical bytes, allocated bytes (hard links
count once), logical duplicate bytes, and reachable/unreachable CAS objects.

## Recoverable garbage collection

GC never starts with deletion. It uses a root-stamped plan and quarantine:

```bash
# 1. Write <ara>/gc/plans/<plan-id>.json.
xscientist ara gc --ara <ara> --grace-seconds 2592000

# 2. Revalidate the exact root set, then move candidates into quarantine.
xscientist ara gc --apply <ara>/gc/plans/<plan-id>.json

# 3a. Recover if the plan was wrong.
xscientist ara gc --restore <ara>/gc/quarantine/<plan-id>/receipt.json

# 3b. Or explicitly purge after the second grace period.
xscientist ara gc --purge <ara>/gc/quarantine/<plan-id>/receipt.json \
  --purge-grace-seconds 2592000
```

If any metadata reference or pin changes after planning, apply is refused.
Plans and receipts are excluded from root discovery so they cannot
accidentally keep their own candidates alive.

## Non-destructive compaction

Legacy artifacts can be migrated without rewriting their history:

```bash
xscientist ara compact --ara <old-ara> --dest <new-ara>
```

The successor uses canonical edges, reference-only claims, CAS-backed verify
output, and a rebuilt semantic catalog. The original lock and revision files move to `legacy/` inside
the successor, while the source ARA is untouched. The new manifest records
`provenance.supersedes_manifest_hash` and receives a fresh lock.

## Retention rule

Relationships are permanent; payloads are tiered. Published claims,
confirmatory evidence, baselines, counter-evidence, active fork ancestors,
signed verification results, and explicit pins must remain reachable. Derived
views and unreferenced CAS objects may be regenerated, archived, or collected
according to deployment policy.
