# Local Research Git

XScientist can record scientific progress in a normal local Git repository.
No GitHub account, remote, daemon, or server is required. A remote can be added
later without rewriting any local commit, branch, or tag.

The design separates three concerns:

| Layer | Stores | Why |
|---|---|---|
| Git | Questions, hypotheses, code, compact metrics, claim/evidence links, checkpoints, and content hashes | Small, diffable scientific history |
| ARA | Exploration DAG, provenance, verification state, and reproducibility metadata | Scientific semantics |
| Local CAS | Datasets, models, binaries, full logs, and other immutable payloads | Complete storage without inflating Git |

XScientist never creates a remote and never pushes automatically. Every Git
mutation is scoped to the research repository and uses an explicit file
whitelist; it never runs `git add -A`.

## Start a standalone research repository

```bash
xscientist research init ./my-research \
  --question "Does retrieval-guided reflection improve factual accuracy?" \
  --policy milestone

cd my-research
xscientist research status
git log --oneline --graph
```

Initialization creates `research.yaml`, `question.md`, a safety-oriented
`.gitignore`, local ARA/CAS directories, and an initial `research(init)`
checkpoint commit. If Git has no configured identity, the repository receives
the local-only fallback `XScientist <xscientist@localhost>`; pass
`--git-user-name` and `--git-user-email` to choose an explicit identity.

## Record scientific progress

```bash
# Add or edit hypotheses/h1.json first.
xscientist research checkpoint \
  --stage preregister \
  --subject "lock H1 and its falsifier" \
  --summary "Prospective metric, baseline, seed, and stopping rule."

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

Checkpoint commits contain machine-readable trailers:

```text
Research-Checkpoint: rcp-...
Research-Stage: experiment
Research-State: completed
Research-Event: sha256:...
ARA-Manifest: sha256:...
Reproduce: xscientist ara verify ...
```

The current commit SHA is intentionally not embedded in its own tree. Git is
the outer identity; manifest/event hashes in the commit trailers provide the
reverse lookup without a self-referential hash.

## Inspect, compare, and reproduce

```bash
xscientist research log --limit 20
xscientist research show HEAD
xscientist research diff HEAD~1 HEAD

# Read the reproduction closure without executing anything.
xscientist research reproduce HEAD --json

# Materialize an exact detached worktree and hydrate its bound CAS objects.
xscientist research reproduce HEAD --dest ../reproduce-head

# Explicit opt-in: execute the checkpoint's single-line command without a shell.
xscientist research reproduce HEAD --dest ../reproduce-run --execute
```

`reproduce` verifies the checkpoint content hash and resolves object pointers
from the selected commit, not from the latest tree. It therefore continues to
work when later commits add or remove pointers, provided the referenced local
CAS objects remain available.

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

The object command hashes the payload, hard-links it into `.ara-store/` when
possible (copying only when necessary), and writes a small pointer under
`research-objects/`. Checkpoints bind the pointer hash; the payload itself is
ignored by Git.

Secrets such as `.env`, credentials, private keys, and token files are denied
both as direct Git files and as CAS logical paths.

## Offline backup without a server

```bash
xscientist research bundle \
  --profile reproduce \
  --dest ../my-research-backup.tar.gz
```

Profiles:

| Profile | Contents |
|---|---|
| `index` | `repository.gitbundle` plus object pointers |
| `reproduce` | Git history, pointers, and the complete referenced CAS closure |
| `audit` | Currently the same durable closure as `reproduce`, reserved for future audit-only additions |

Bundling refuses dirty repositories and missing objects by default. The
archive includes `bundle.manifest.json` with hashes, sizes, HEAD, profile, and
a completeness verdict.

## Integrate with an XScientist project run

```bash
xscientist project my_project \
  --topic topic.md \
  --research-git local \
  --git-checkpoint-policy milestone
```

The project directory becomes its own nested/local Git repository. The
`milestone` policy records initialization, experiment outcomes, and final
paper/shortlist state. `stage` additionally records ideation. `manual` creates
only the initialization commit; later checkpoints are operator-controlled.

Use `--research-git-strict` when checkpoint failure must fail the research
command. Without it, expensive research outputs are preserved and Git errors
are surfaced as warnings for later repair.

## Branches and later GitHub synchronization

Use commits for progress within one line of inquiry and branches only for real
scientific divergence:

```bash
git switch -c hypothesis/retrieval-reflection
git switch -c method/graph-rag
```

When a remote becomes available:

```bash
git remote add origin <url>
git push -u origin --all
git push origin --tags
```

These are ordinary Git commands initiated by the operator. XScientist keeps
`auto_push: false` as a schema-enforced invariant.
