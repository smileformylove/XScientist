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
| Research VCS | Typed objects, relations, semantic stage, research lines, commits, gates, and provenance | Stable public scientific history |
| ARA | Exploration DAG, provenance, verification state, and reproducibility metadata | Scientific semantics |
| Local CAS | Datasets, models, binaries, full logs, and other immutable payloads | Complete storage without inflating Git |
| Git adapter | Durable commit graph and optional interoperability | Replaceable implementation detail |

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
self-referential hash.

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

For ecosystem exchange, export one committed ref without exposing payloads by
default:

```bash
xscientist research export --repo . --ref HEAD --dest ../research-export
```

The export contains a hash-bound manifest plus RO-Crate, W3C PROV-JSON, CWL,
DVC, and MLflow adapter files. Repeat `--format` to select a subset. Scientific
payloads enter RO-Crate only with the explicit `--include-payloads` flag.

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
