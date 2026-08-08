# Beginner Workflow, Scientific DAG, and Platform Adapters

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

`start` creates the question and falsifiable hypothesis in one initial
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
- agent candidates, independent agent evaluations, promotions, deployments,
  and rollback decisions;
- optional ARA experiment-search nodes and their parent/child evolution paths.

Attach one or more ARA roots when detailed experiment exploration should be
shown alongside the compact Research VCS argument:

```bash
xscientist research dag \
  --ara ./ara/20260808_example \
  --output ./research-dag
```

When a Research Object's `ara_manifest_hash` matches an ARA `manifest.lock`,
the DAG adds an explicit `anchors` edge from the ARA experiment leaves to that
object. Missing targets and graph cycles block the DAG integrity verdict.

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
