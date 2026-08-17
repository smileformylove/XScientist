# Adapter cookbook

XScientist has two integration directions:

- export a committed, hash-bound package through an explicit adapter;
- ingest a tool receipt as unverified evidence tied to an experiment attempt.

Neither direction can silently verify a scientific claim.

## MLflow

Export the committed state in MLflow-oriented and provenance formats:

```bash
xscientist research adapter sync filesystem \
  --repo ./study --ref HEAD --dest ./exchange/mlflow-study \
  --format mlflow --format prov-json
```

To bring a completed MLflow run back, create a `tool_evidence` receipt with the
tool version, run ID, summary metrics, and content hashes for retained
artifacts. Then bind it to the exact attempt:

```bash
xscientist research ingest mlflow-evidence.json \
  --repo ./study \
  --attempt @latest:experiment_attempt \
  --supports @latest:hypothesis
```

Do not put tracking tokens, artifact URLs containing credentials, or local file
paths in the receipt.

## DVC

Publish a committed exchange package suitable for a DVC project:

```bash
xscientist research adapter sync filesystem \
  --repo ./study --dest ./exchange/dvc-study \
  --format dvc --format ro-crate --format prov-json
```

Keep large datasets in the platform's storage. Research VCS records immutable
dataset/artifact hashes and portable references rather than copying a secret or
machine-specific cache path into scientific history.

## Notebook

Use the public API to record a notebook-derived observation without shelling
out:

```python
from xscientist import ResearchRepository

repo = ResearchRepository("./study")
attempt_id = repo.resolve("@latest:experiment_attempt", kind="experiment_attempt")
observation = repo.record(
    "observation",
    {"measurement": "held-out accuracy", "metrics": {"accuracy": 0.82}},
    state="completed",
    relations=[{"type": "generated_by", "target": attempt_id}],
    actor={"actor_id": "notebook-recorder", "authority": "recorder"},
)
repo.checkpoint(
    stage="evidence",
    subject="Record held-out notebook observation",
    summary=observation.object_id,
)
```

The notebook is a recorder, not an independent evaluator.

## Custom platform adapter

Register an entry point:

```toml
[project.entry-points."xscientist.research_adapters"]
lab-platform = "lab_adapter:LabPlatformAdapter"
```

Implement `descriptor`, `probe()`, and `publish(package_root, destination,
options=...)` using `ResearchAdapterDescriptor`. Test the object with
`validate_research_adapter()`, then run:

```bash
xscientist research adapter list
xscientist research adapter doctor lab-platform
xscientist research adapter sync lab-platform --repo ./study --dest lab://study-42
```

`publish` receives only a temporary committed exchange package. Returned data
must not contain credentials; XScientist rejects a sensitive result rather than
persisting it into the adapter receipt.
