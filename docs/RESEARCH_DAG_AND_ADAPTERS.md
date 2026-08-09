# Beginner Workflow, Scientific DAG, and Platform Adapters

`xscientist project ... --autopilot` exports the unified DAG automatically at
the end of a successful project. Its self-contained browser is written to
`<output_root>/views/<project>/research-dag/research-dag.html`, outside the
Research VCS working tree, so refreshing a view cannot alter scientific
history. Machine-synthesized insights appear as draft, unverified claim nodes
linked to run evidence and hold gates.

This layer is designed for people who understand their question but do not
already know Git, provenance standards, autonomous-agent governance, or the
XScientist object model.

## Start with three sentences

You need only a question, a proposed answer, and an observation that would show
the proposed answer is wrong:

```bash
xscientist research start ./my-study \
  --question "Does spaced practice improve one-week recall?" \
  --hypothesis "Spaced practice improves one-week recall." \
  --falsifier "Recall is unchanged or worse than the fixed baseline." \
  --actor human:alice

cd my-study
xscientist research guide --lang en
```

`start` creates the question, a locked research goal, and falsifiable hypothesis in one initial
checkpoint. `guide` inspects actual repository state and prints one safe next
step (or an explicit exploratory/confirmatory choice), why it matters, and a
copy/paste command. It never asks a beginner to
construct a Research Object ID: commands use selectors such as
`@latest:hypothesis`.

The guide deliberately does not claim that every study must be confirmatory.
It explains the choice between exploration and a locked preregistration. It
also warns when a study has accumulated supporting evidence without any
recorded refutation or contradiction attempt.

Exploratory users can create a typed plan without writing JSON:

```bash
xscientist research plan @latest:hypothesis \
  "Compare spaced practice with the fixed baseline" \
  --test "Measure one-week recall under the same sampling policy"
```

Chinese guidance is built in:

```bash
xscientist research guide --lang zh
```

## See the whole scientific argument

```bash
xscientist research dag --output ./research-dag
open ./research-dag/research-dag.html       # macOS
# xdg-open ./research-dag/research-dag.html # Linux
# start ./research-dag/research-dag.html    # Windows
```

The command writes deterministic `research-dag.json` and a self-contained,
offline `research-dag.html`. The browser requires no server, CDN, account, or
JavaScript package installation. It is searchable, keyboard accessible, and
can filter by scientific object kind and verification level.

The graph combines:

- questions, hypotheses, research plans, and locked preregistrations;
- successful, failed, timed-out, and cancelled experiment attempts;
- supporting, refuting, and contradictory evidence;
- claims, independent reviews, deterministic gates, and reproductions;
- exact Research VCS decision-context snapshots and their evidence/memory
  sources;
- estimands, effect estimates, explicit inferences, warrants, assumptions,
  methods, challenges, and source correction/retraction events;
- agent candidates, independent agent evaluations, promotions, deployments,
  and rollback decisions;
- committed ARA experiment-search nodes and their parent/child evolution paths,
  discovered from the selected Research Git ref;
- optional external ARA roots supplied explicitly for local comparison.

Context is visible rather than hidden inside prompts. A Research VCS
`context_snapshot` has dashed context edges from every immutable source it
consumed; reviews, deterministic gates, and agent evaluations have a context
edge from that snapshot. ARA `context_pack_refs` appear as content-addressed
context nodes connected to the experiment that consumed them. When an ARA
ContextPack hash is propagated into a Research Object, the graph adds the
cross-system edge too. This makes the chain
`memory/evidence -> context -> experiment/decision -> claim` inspectable by a
human or an agent.

New context snapshots also carry a retrieval receipt with the complete
candidate set, deterministic ranks, score semantics, selection/rejection
reasons, algorithm hash, and summary-transform lineage. A token budget can hide
a readable summary, but it cannot erase the candidate or its immutable hash.
See [Research Protocol v2](RESEARCH_PROTOCOL_V2.md) for the compatibility and
extension rules.

Committed ARAs whose current or prior legal manifest revision is referenced by
a Research Object are discovered automatically. The manifest, revision history,
and exploration graph are all read from the exact Git commit selected by
`--ref`, so a historical DAG cannot silently mix an old Research Git state with
the current ARA worktree. New checkpoints also bind the canonical
`exploration_graph.json` hash; a raw graph edit outside the checkpointed
transition blocks ARA verification edges. Older checkpoints remain explicitly
reported as `commit_bound` because the Git commit still fixes their graph bytes.
When objects at the selected ref bind older graph revisions, the projection
rehydrates the matching checkpointed snapshots from reachable Git history and
shows them as separate versioned ARA sources. Thus earlier evidence stays on
the graph it actually observed while the newest exploration remains visible.

Attach one or more additional ARA roots only when an external ARA should be
shown alongside the committed Research VCS argument:

```bash
xscientist research dag \
  --ara ./ara/20260808_example \
  --output ./research-dag
```

New Research Objects bind the pair `ara_manifest_hash` plus
`ara_exploration_graph_hash`. The DAG requires both values to match the same
versioned ARA source before it adds an `anchors` edge from the referenced
experiment node, or otherwise from that snapshot's leaves, to the object.
Manifest-only legacy objects still match any legal manifest revision. External
worktree ARAs without a checkpoint graph binding are shown as lineage rather
than verification. Missing or conflicting bindings, missing targets, and graph
cycles block the DAG integrity verdict.

The selected Git ref controls all three inputs: Research Objects, committed ARA
snapshots, and ContextPack references. A historical graph therefore retains
the failures and alternatives known at that commit while excluding later
memory. A context node is `replayable` only when its context, source-closure,
selection-policy, and memory hashes recompute and its declared hard closure is
complete. An old decision without a context binding stays readable with a
`legacy_decision_context_unbound` warning; a new decision declaring context as
required is blocked when that binding is missing or invalid.

## What the verification colors mean

The browser uses five deliberately narrow levels:

| Level | Meaning |
|---|---|
| `recorded` | The immutable object is valid, but required scientific links are missing. |
| `traceable` | The object can be followed to its immediate scientific sources. |
| `replayable` | Code, data, environment, dependency, and measurement identities needed for replay are present. |
| `verified` | The applicable independent gate, authority, receipt, and reproduction checks passed. |
| `contested` | Refuting or contradictory evidence targets this object; it must remain visible even if other checks pass. |

Selecting a node shows every individual check and its `trace`, `replay`, or
`verify` layer. A green node is therefore not a generic “truth” badge. It means
that the explicit checks applicable to that recorded object passed. Physical
custody, institutional review, wet-lab observations, hardware-backed keys, and
external timestamping still require external authorities.

For a shareable metadata-only graph that hides statements and result summaries:

```bash
xscientist research dag --metadata-only --json
```

The HTTP service uses metadata-only mode by default:

```text
GET /v1/projects/{project}/research/dag?ref=HEAD
```

Clients must explicitly set `include_summaries=true` to receive short local
content summaries.

## Use the same repository from Python

```python
from xscientist import ResearchRepository

repo = ResearchRepository("./my-study")
print(repo.guide(language="en")["next_steps"][0])
graph = repo.dag(disclose_summaries=False)
repo.export_dag("./research-dag")
```

The JSON DAG conforms to `research_dag.schema.json`, is content-addressed, and
uses the same committed ref as closure auditing. It is suitable for notebooks,
desktop apps, static sites, and read-only dashboards.

## Connect tools and platforms without changing core code

Existing built-in exchange formats remain available:

- RO-Crate for packaged research context;
- W3C PROV-JSON for provenance systems;
- CWL for workflow engines;
- DVC for data/model pipeline projects;
- MLflow for run tracking.

The adapter layer publishes this hash-bound exchange package to a selected
platform. Adapter discovery does not import third-party code. A plugin is
loaded only after its exact name is used with `doctor` or `sync`:

```bash
xscientist research adapter list
xscientist research adapter doctor filesystem
xscientist research adapter sync filesystem \
  --dest /mounted/shared-study \
  --format ro-crate --format prov-json
```

The built-in `filesystem` adapter works with local directories, mounted cloud
drives, shared volumes, and network filesystems on macOS, Linux, and Windows.
It publishes atomically and refuses to overwrite an existing destination.

External tools can also send evidence into the repository without receiving
verification authority. Write a `tool_evidence.schema.json` receipt:

```json
{
  "schema_version": "xscientist.tool-evidence.v1",
  "tool": {"name": "mlflow", "version": "3.0"},
  "run_id": "run-123",
  "result": "Accuracy reached 0.82.",
  "metrics": {"accuracy": 0.82},
  "artifact_hashes": ["sha256:0000000000000000000000000000000000000000000000000000000000000000"]
}
```

Then bind it to the originating attempt and scientific direction:

```bash
xscientist research ingest tool-evidence.json \
  --attempt @latest:experiment_attempt \
  --supports @latest:hypothesis
```

The file path and raw file are not stored. Its canonical hash, tool/version,
run ID, metrics, result, and artifact hashes enter one completed evidence node
with `recorder` authority. The node remains unverified until an independent
review and reproduction pass; importing from MLflow, DVC, a notebook, an ELN,
or a custom instrument bridge cannot bypass that boundary.

Third-party packages register an entry point in `pyproject.toml`:

```toml
[project.entry-points."xscientist.research_adapters"]
my-platform = "my_package.adapter:MyPlatformAdapter"
```

The loaded object implements the versioned public contract:

```python
from pathlib import Path
from xscientist.research_adapters import ResearchAdapterDescriptor

class MyPlatformAdapter:
    descriptor = ResearchAdapterDescriptor(
        name="my-platform",
        version="1.0.0",
        description="Publish to My Platform",
        capabilities=("publish",),
        destination_kinds=("workspace-uri",),
        source="entry-point:my-package",
    )

    def probe(self):
        return {"ok": True, "requirements": [], "errors": []}

    def publish(self, package_root: Path, destination: str, *, options):
        # Upload only files from package_root. Never return credentials.
        return {"status": "published", "remote_id": "study-123"}
```

Use `validate_research_adapter()` in the plugin's tests. Every successful sync
returns `research_adapter_receipt.schema.json` containing the adapter version,
committed ref, export-manifest hash, payload-disclosure flag, platform result,
timestamp, and canonical receipt hash. Credentials belong in the platform's
normal secret store and must never appear in the receipt.

## Platform matrix

| Surface | Best use | Mutation |
|---|---|---|
| CLI `start`, `guide`, `dag` | Individual researchers and offline work | Explicit local commands |
| Python `ResearchRepository` | Notebooks, desktop applications, automation | Caller controlled |
| HTTP `/research/dag` | Web apps and organization dashboards | Read-only |
| Standard exports | RO-Crate/PROV/CWL/DVC/MLflow consumers | Writes a new exchange directory |
| Adapter entry points | Cloud platforms, ELNs, trackers, object stores | Explicit named `sync` only |
| Tool evidence receipt | MLflow/DVC/notebooks/ELNs/instruments | Creates unverified evidence only |

No adapter is allowed to weaken Research VCS validation. Platform publication
is a downstream operation over a committed, hash-bound exchange package; it
does not silently mark a claim verified or mutate scientific history.
