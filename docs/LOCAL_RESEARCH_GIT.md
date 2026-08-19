# Native Research Version Control

XScientist versions scientific meaning directly: questions, hypotheses,
preregistrations, experiment attempts, evidence, claims, reviews, gate
decisions, manuscripts, reproductions, and agent-evolution candidates. No
GitHub account, remote, daemon, or server is required. Git is the current local
persistence adapter; public operations and identifiers remain Research VCS
semantics.

The design separates three concerns:

| Layer | Stores | Why |
|---|---|---|
| Research VCS | Typed objects, exact decision-context snapshots, relations, semantic stage, research lines, commits, gates, and provenance | Stable public scientific history |
| ARA | Exploration DAG, consumed ContextPacks, provenance, verification state, and reproducibility metadata | Scientific semantics |
| Local CAS | Datasets, models, binaries, full logs, and other immutable payloads | Complete storage without inflating Git |
| Git adapter | Durable commit graph and optional interoperability | Replaceable implementation detail |

For everyday use, the small facade is usually enough:

```bash
xscientist history list ./my-research
xscientist history save ./my-research -m "record measurement decision"
xscientist history rollback ./my-research --commit HEAD
xscientist audit ./my-research --level trace
```

Rollback is preview-only by default and appends a reversal checkpoint only with
`--apply`; it never rewrites history. The rest of this page documents the full
protocol surface for advanced workflows and integrations.

XScientist never creates a remote and never pushes automatically. Every backend
mutation is scoped to the research repository and uses an explicit privacy and
file policy; it never stages the whole working tree implicitly.

## Start a standalone research repository

```bash
xscientist research init ./my-research \
  --question "Does retrieval-guided reflection improve factual accuracy?" \
  --policy milestone

cd my-research
xscientist research status
xscientist research log
```

Initialization creates `research.yaml`, `question.md`, a safety-oriented
`.gitignore`, local ARA/CAS directories, and an initial `research(init)`
checkpoint commit. If Git has no configured identity, the repository receives
the local-only fallback `XScientist <xscientist@localhost>`; pass
`--git-user-name` and `--git-user-email` to choose an explicit identity.

## Record typed scientific progress

```bash
# Record one immutable object. Repeating identical content is idempotent.
xscientist research record hypothesis \
  --data '{"statement":"H1","falsifier":"no improvement over baseline"}'

# Select exact changes and create one atomic scientific transition.
xscientist research stage --all
xscientist research checkpoint --staged \
  --stage ideation \
  --subject "record H1 and its falsifier"

# Bind a completed experiment and ARA manifest.
xscientist research checkpoint \
  --stage experiment \
  --subject "complete baseline n12" \
  --node n12 \
  --ara ara/run-001 \
  --reproduce "xscientist ara verify --ara ara/run-001 --limit 3"

# Bind claims after evidence review.
xscientist research checkpoint \
  --stage evidence \
  --subject "bind c12 to confirmatory and counter evidence" \
  --claim c12
```

Research commits contain machine-readable trailers in the current adapter:

```text
Research-Checkpoint: rcp-...
Research-Stage: experiment
Research-State: completed
Research-Event: sha256:...
ARA-Manifest: sha256:...
Reproduce: xscientist ara verify ...
```

The backend commit SHA is intentionally not embedded in its own tree. Research
object/checkpoint content hashes remain the scientific identities and avoid a
self-referential hash. Each new checkpoint ARA reference also stores the
canonical `exploration_graph.json` hash in its checkpoint JSON. This binds the
node/edge projection to the same scientific transition while retaining
read compatibility with older commit-bound checkpoints. ARA bindings carry
forward across later checkpoints until that ARA is explicitly changed or
passed with `--ara`; an unrelated checkpoint therefore cannot silently accept
a graph that was edited through raw Git.

## Inspect, compare, and reproduce

```bash
xscientist research log --limit 20
xscientist research show HEAD
xscientist research diff HEAD~1 HEAD
xscientist research diff HEAD~1 HEAD --deep
xscientist research fsck
xscientist research audit --level trace
xscientist research audit --level replay
xscientist research audit --level verify

# Resolve IDs without copying them by hand.
xscientist research objects @latest:hypothesis

# Inspect exactly what a decision would see, including failures and prior gates.
xscientist research context @latest:hypothesis \
  --decision-kind next_experiment \
  --selected revise_method \
  --option revise_method \
  --option 'repeat_failed_setup=the retained attempt already failed' \
  --rationale 'use the failed run as negative knowledge'

# Persist that context as an immutable Research Object and checkpoint.
xscientist research context @latest:evidence \
  --decision-kind evidence_triage \
  --selected hold --option hold \
  --option 'promote=independent reproduction is missing' \
  --record

# Reconstruct context from an old commit; selectors resolve inside that commit.
xscientist research context @latest:hypothesis --ref HEAD~3 --json

# Safe maintenance and recovery operations.
xscientist research branch challenge/h1 -m challenge/accuracy
xscientist research branch challenge/accuracy -d
xscientist research restore HEAD~1 claims/result.md
xscientist research revert <checkpoint-commit>

# Read the reproduction closure without executing anything.
xscientist research reproduce HEAD --json

# Materialize an exact detached worktree and hydrate its bound CAS objects.
xscientist research reproduce HEAD --dest ../reproduce-head

# Explicit opt-in: execute the checkpoint's single-line command without a shell.
xscientist research reproduce HEAD --dest ../reproduce-run --execute \
  --environment-policy strict
```

`reproduce` verifies the checkpoint, pointer, object size, and object SHA-256,
then copies CAS payloads into the detached worktree so experiments cannot
mutate the store through hard links. It resolves pointers from the selected
commit, not from the latest tree. Every new checkpoint also binds a compact,
secret-free environment receipt containing Python/platform identity and hashes
of supported dependency lock files. `warn` is the compatibility default;
`strict` refuses runtime or lock drift; `ignore` records but does not gate it.
Every inspection, materialization, and rerun produces a
`reproduction_receipt.schema.json` receipt. It stores hashes of the command and
stdout/stderr rather than duplicating their contents. A detached worktree gets
the receipt under `.xscientist/reproductions/`. An independent verifier may
bind a passing receipt back to typed objects with `reproduce --record
--verified --verifier ... --reproduces ...`.

`research audit` answers a different question from `fsck`. `fsck` verifies
storage integrity; `audit` checks scientific sufficiency without disclosing
payloads:

| Level | Required closure |
|---|---|
| `trace` | claim → evidence → attempt → plan; locked preregistration for confirmatory work |
| `replay` | trace plus immutable code, data, environment, dependency-lock/container-recipe, and measurement/ARA identities |
| `verify` | replay plus an independently reviewed deterministic gate, verified claim, and hash-recomputed successful reproduction receipt |

`verify` means that the selected ref satisfies XScientist's local protocol
closure. It does not replace signatures, external custody, peer review, or a
third-party scientific attestation.

`research context` answers a third question: “which exact evidence and memory
did this decision consume?” Its audit closure retains full object IDs and hashes
for supporting and negative evidence, failed attempts, prior reviews/gates, and
the latest related context-chain link. The agent-facing working set is a
separate, budgeted projection. It ranks the effective frontier before
superseded history, reserves current evidence or active contradiction and a
relevant prior decision, and reports `decision_usable=false` when those required
semantics do not fit. Recorded
reviews and gates bind this snapshot using a `decision_context` DAG edge and a
matching `context_hash`; closure verification fails closed if a required
snapshot is missing or changed. Historical `--ref` reads objects and resolves
`@latest:<kind>` at that ref, so old decisions cannot accidentally see today's
worktree memory. Use `research context ... --json` for the full auditable
snapshot and `research context ... --prompt` for the bounded source-bound view
that should be injected into an agent.

For ecosystem exchange, export one committed ref without exposing payloads by
default:

```bash
xscientist research export --repo . --ref HEAD --dest ../research-export
```

The export contains a hash-bound manifest plus Process Run RO-Crate, W3C
PROV-JSON, CWL, DVC, MLflow, OpenLineage, Croissant, and Nanopublication files.
Repeat `--format` to select a subset. Scientific payloads enter RO-Crate only
with the explicit `--include-payloads` flag.

The audit is a derived index over IDs and hashes. Raw logs, datasets, and full
ARA contents remain stored and drill-downable, but are not loaded merely to
answer “what supports this claim?”.

`fsck` verifies the checkpoint ancestry DAG, ARA manifest bindings, pointer
schemas and hashes, configured CAS paths, object sizes, and payload hashes.
`--no-objects` skips the potentially expensive payload scan and reports that
choice as a warning.

## Store large evidence without Git inflation

Large/binary files are denied by the Git whitelist. Register them in the local
CAS instead:

```bash
xscientist research object add ./raw/results.parquet \
  --logical-path data/results.parquet

xscientist research checkpoint \
  --stage evidence \
  --subject "register immutable result table"
```

The object command streams one independent immutable snapshot into
`.ara-store/`, hashing during the copy, and writes a small pointer under
`research-objects/`. Source-file edits after registration cannot mutate CAS.
Checkpoints bind the pointer hash; the payload itself is ignored by Git.

Secrets such as `.env`, credentials, private keys, and token files are denied
both as direct Git files and as CAS logical paths.

## Offline backup without a server

```bash
xscientist research bundle \
  --profile reproduce \
  --dest ../my-research-backup.tar.gz

xscientist research bundle verify ../my-research-backup.tar.gz
xscientist research bundle restore ../my-research-backup.tar.gz \
  --dest ../restored-research
```

Profiles:

| Profile | Contents |
|---|---|
| `index` | `repository.gitbundle` plus object pointers |
| `reproduce` | Git history, pointers, and the complete referenced CAS closure |
| `audit` | Currently the same durable closure as `reproduce`, reserved for future audit-only additions |

Bundling refuses dirty repositories and missing objects by default. The
archive includes `bundle.manifest.json` with hashes, sizes, HEAD, profile, and
a completeness verdict. Verification rejects duplicate, hidden, non-regular,
unsafe, missing, size-mismatched, or hash-mismatched members. Restore writes
only declared regular members, removes the temporary bundle remote, restores
the exact HEAD, and runs `fsck` before publishing the destination directory.

## Integrate with an XScientist project run

```bash
xscientist project my_project \
  --topic topic.md \
  --checkpoint-policy milestone
```

Research VCS is enabled by default; use `--research-vcs off` only for an
explicitly history-free run. The project directory gets a local adapter. The
`milestone` records initialization, ideation/planning, preregistration,
experiment outcomes, evidence/review/claim transitions, paper state,
self-evolution, merges, and releases. `stage` checkpoints every requested
stage. `manual` creates only the initialization commit; later checkpoints are
operator-controlled.

When ARA finalization succeeds, the project flow automatically projects each
run into one compact evidence object and one hash-only Research VCS claim per
ARA claim. It records the manifest hash and source IDs instead of copying claim
text, node bodies, or logs into Git.

Autopilot runs also project each evidence-bound item from
`04_logs/insight_report.json` as a draft claim. It preserves exact run evidence
selectors, calibrated confidence, unresolved rivals, and the next
high-information experiment. Internal agent review never promotes the claim to
`verified`; the independent-verification hold gate remains authoritative.

Use `--research-vcs-strict` when checkpoint failure must fail the research
command. Without it, expensive research outputs are preserved and adapter errors
are surfaced as warnings for later repair.

## Research lines, semantic merge, and provenance

Use commits for progress within one line of inquiry and branches only for real
scientific divergence:

```bash
xscientist research branch hypothesis/retrieval-reflection --switch
xscientist research branch challenge/graph-rag
xscientist research switch challenge/graph-rag
xscientist research merge hypothesis/retrieval-reflection --preview
xscientist research merge hypothesis/retrieval-reflection
xscientist research decide contradiction \
  --name alternate-mechanism --contradictory-evidence
xscientist research tree
xscientist research blame <research-object-id>
```

Checkpoint sequence numbers are branch-local. When Research VCS merges two
scientific branches, the next checkpoint records both parent checkpoint hashes
while retaining the first parent in `previous_checkpoint_hash` for v1 reader
compatibility. Checkpoint creation, object registration, and bundle snapshots
share a repository lock; a failed Git commit removes only the new checkpoint
files and this attempt's staged entries, preserving user research files.

The native merge preflight blocks file conflicts, opposing support/refutation,
incompatible locked preregistrations, differing metric definitions, and
ungated agent candidates entering `main`/`stable`. A successful merge retains
both scientific parents. Every conflict has a stable ID and resolution guidance.
Opposing evidence alone may be explicitly retained with
`--preserve-conflicts`; this writes a rejected deterministic `hold` gate that
binds the target and both evidence sets. All other conflict classes remain
blocked. No side is overwritten and no contested claim is promoted.

`research decide` is the read-only policy boundary used by autonomous agents:
it recommends checkpoints for material terminal/milestone state, forks for an
independent hypothesis, incompatible method, contradictory interpretation,
replication, or agent candidate, and merges only after a clean preflight. Its
stable decision ID, inputs, reasons, and commands form the decision trace.
`research tree` derives a payload-free long-term technology tree from immutable
objects and relations, including open/contested frontier nodes, branch heads,
topological order, cycles, and missing relation targets.

For a human-facing view of the complete scientific argument, use:

```bash
xscientist research guide --lang en
xscientist research dag --ara ./ara/<run> --output ./research-dag
```

The unified DAG is distinct from the compact technology tree. It projects every
Research VCS object, support/refutation relation, independent review, gate,
reproduction, and agent-evolution transition, then optionally connects detailed
ARA experiment nodes through `manifest.lock`. Selecting a node reveals its
trace, replay, and verify checks. The output is both schema-valid JSON and a
self-contained offline HTML browser.

For a deeper strategy review, use:

```bash
xscientist research program template --output deep-research.json
xscientist research program review
xscientist research program review --record
xscientist research program claim @latest:claim
```

These commands add or inspect competitive hypotheses, discriminating
predictions, information-value experiment priorities, anomalies, mechanisms,
quality audits, and transfer boundaries. They use the same immutable object
store and branch history; [the deep research protocol](DEEP_RESEARCH_PROTOCOL.md)
defines their gates and automation boundary.

Committed exchange packages can be published through an explicit adapter:

```bash
xscientist research adapter list
xscientist research adapter doctor filesystem
xscientist research adapter sync filesystem \
  --dest ../shared-study --format ro-crate --format prov-json
```

Third-party adapters are discovered through the
`xscientist.research_adapters` entry-point group and are never imported by the
listing command. See `docs/RESEARCH_DAG_AND_ADAPTERS.md` for the versioned
contract and platform matrix.

## Optional Git interoperability

When an operator explicitly wants a Git remote, the current adapter can use
ordinary Git commands:

```bash
git remote add origin <url>
git push -u origin --all
git push origin --tags
```

These are ordinary Git commands initiated by the operator. XScientist keeps
`auto_push: false` as a schema-enforced invariant.
