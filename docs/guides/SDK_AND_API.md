# XScientist SDK and API

XScientist exposes one public package, `xscientist`. The historical
`ai_scientist` package contains the workflow implementation, while
wheel-level modules preserve legacy compatibility. Application consumers
should depend on `xscientist`; contributors may work in `ai_scientist`.

## Installation profiles

The stable package is available from PyPI. Unreleased Research VCS changes must
be installed from the branch that contains them until the next release:

```bash
pip install xscientist
pip install "xscientist[research,openai]"
pip install "xscientist[research,openai,service]"

pip install "xscientist @ git+https://github.com/smileformylove/XScientist.git@main"
pip install "xscientist[research,openai] @ git+https://github.com/smileformylove/XScientist.git@main"
pip install "xscientist[research,openai,service] @ git+https://github.com/smileformylove/XScientist.git@main"

# From a local clone:
pip install -e ".[research,openai,service,dev]"
```

Replace `openai` with `anthropic`, `zhipu`, `bedrock`, `vertex`, or
`openai-compatible` to install only the selected provider client. Add `ml`,
`pdf-layout`, or `service` only when the application needs that capability.
The `full` extra remains available as a backwards-compatible all-in-one
profile.

Run `xscientist git doctor` after installation to verify the local Research VCS
adapter and its safe-merge capabilities.

## Package boundary

```text
xscientist/                 Public, versioned integration surface
├── client.py               Python SDK, subprocess isolation, read-only views
├── models.py               Stable request/result data models
├── cli.py                  Unified `xscientist` command
├── service.py              Optional FastAPI service
├── research_vcs.py         Backend-independent scientific version-control API
├── research_lifecycle.py   Evidence-gated research transitions
├── research_evolution.py   Agent candidate promotion and rollback
├── research_commands.py    One-command researcher workflows
├── research_journey.py     Beginner start/guide state machine
├── research_dag.py         Unified evidence/evolution graph and browser
├── research_adapters.py    Versioned external platform plugin contract
├── research_tools.py       Schema-bound external evidence ingestion
└── entrypoints.py          Compatibility workflow dispatch

ai_scientist/               Internal workflow implementation
├── apps/                   Installed project/batch/daemon/manager apps
├── protocol/               Stable on-disk research protocol
├── resources/              Packaged configs and resource lookup
├── treesearch/             Experiment search engine
└── utils/                  Internal workflow components

wheel root modules          Thin legacy aliases to ai_scientist.apps
```

Public compatibility follows semantic versioning for symbols exported by
`xscientist.__all__`. Internal modules can evolve more quickly.

The four primary workflows have one implementation each:

| Workflow | Installed/source command | Internal application | Wheel compatibility alias |
|---|---|---|---|
| Single project | `xscientist project` | `ai_scientist.apps.project` | `run_project.py` |
| Batch generation | `xscientist batch` | `ai_scientist.apps.batch` | `continuous_paper_generator.py` |
| Long-running daemon | `xscientist daemon` | `ai_scientist.apps.daemon` | `continuous_research_daemon.py` |
| Research boards | `xscientist manager` | `ai_scientist.apps.manager` | `research_manager.py` |

The compatibility modules are published at the wheel root and alias the
internal applications. They are intentionally absent from a source checkout;
new integrations should use the public SDK or unified CLI.

`scripts/daemon/run_daemon_profile.py`,
`scripts/daemon/run_daemon_rehearsal.py`, `run_stable_daemon.sh`, and
`start_research.sh` are source-checkout operations tools. They depend on the
repository's example profiles and shell layout, so they are intentionally not
installed by the wheel. Use `xscientist daemon` for portable installed usage.

## Python SDK

```python
from xscientist import ProjectRequest, XScientist

client = XScientist(
    output_root="./research-output",
    env={"OPENAI_API_KEY": "..."},
)

request = ProjectRequest(
    project="demo",
    question="Why does the mechanism fail out of distribution?",
    autopilot="balanced",
    allow_synthetic_data=True,
    max_project_tokens=400_000,
    bfts_config="deep",
)

result = client.run_project(request)
if not result.ok:
    raise RuntimeError(result.stderr)
```

Workflows run in a child process. Large research modules, provider SDK state,
and environment mutations therefore do not leak into an embedding web app.

The same client exposes read-only views over its configured `output_root`:

```python
papers = client.list_papers(sort_by="quality", limit=20)
paper = client.get_paper("paper-folder-name")
shortlist = client.shortlist_papers(
    target_venue="iclr",
    require_gate=True,
    top_n=5,
)
submission_board = client.submission_board(
    require_gate=True,
    top_n_per_venue=3,
)
rewrite_board = client.rewrite_board(top_n=10)
```

These methods do not launch models or modify research artifacts. They reuse the
same manager read models as the CLI and are suitable for dashboards, notebooks,
and embedding applications.

Native research version control is also available without launching a model:

```python
from xscientist import ResearchLifecycle, ResearchRepository

repository = ResearchRepository.init(
    "./study",
    question="Does method A improve the fixed baseline?",
)
lifecycle = ResearchLifecycle(repository)

next_step = repository.guide(language="en")["next_steps"][0]
metadata_graph = repository.dag(disclose_summaries=False)
structured_trajectory = repository.trajectory(limit=50)
repository.export_dag("./research-dag")
```

`trajectory()` verifies the exact included checkpoints and returns a bounded,
payload-free projection of typed object identities, relations, actors and
checkpoint-parent edges. Its `projection_hash` is an inspection artifact, not
the stricter publication attestation or an independent authority decision.

Method-discovery contracts use the same local repository from Python:

```python
from xscientist import (
    discovery_contract_template,
    save_discovery_contract,
    save_generalization_assessment,
)

spec = discovery_contract_template()
# Replace every REPLACE_* value, then lock before running experiments.
contract = save_discovery_contract(
    "./study",
    hypothesis_id="@latest:hypothesis",
    spec=spec,
)

assessment = save_generalization_assessment(
    "./study",
    contract_id=contract["object"].object_id,
    results=condition_results,
    evidence_ids=["@latest:evidence"],
)
```

The builders are deterministic and platform-neutral; notebooks, schedulers,
and third-party agents can create the same contract and assessment objects
without shelling out to the CLI.

The CLI provides both research-native and familiar version-control verbs:

```bash
xscientist git doctor
xscientist research hypothesis "H1" --falsifier "no improvement"
xscientist research preregister <hypothesis-id> \
  --dataset benchmark-v1 --metric accuracy --baseline baseline-a \
  --split-file ./splits/benchmark-v1.json \
  --registered-by recorder:lead-researcher
xscientist research experiment "run failed" --status failed
xscientist research review "independent checks passed" \
  --evaluates <evidence-id> --verifier independent-reviewer --decision pass
xscientist research trajectory --repo ./study
xscientist git add -A
xscientist git commit --stage evidence -m "bind result"
```

`--registered-by` is self-reported recorder provenance. It does not authenticate
a `human:` principal or grant independent evaluator/verifier authority.

## Unified CLI

```bash
xscientist info
xscientist start ./study --question "Does X affect Y?" --allow-synthetic-data
xscientist project demo --topic topic.md
xscientist batch --help
xscientist daemon --help
xscientist manager --help
xscientist ara --help
xscientist auth status
xscientist git doctor
xscientist research --help
```

Direct compatibility commands (`xscientist-project`, `xscientist-batch`,
`xscientist-daemon`, and `xscientist-ara`) are also installed.

## HTTP API

```bash
export XSCIENTIST_API_KEY="replace-with-a-secret"
xscientist serve --host 0.0.0.0 --port 8000 --output-root ./research-output
```

FastAPI exposes interactive documentation at `/docs` and the OpenAPI schema at
`/openapi.json`.

Non-loopback bindings fail closed unless an API key is configured. For a
deliberately unauthenticated private network, the operator must pass
`--allow-unauthenticated` explicitly. The health endpoint remains public;
all `/v1` endpoints require the key when configured:

```bash
export XSCIENTIST_API_KEY="replace-with-a-secret"
xscientist serve --host 0.0.0.0 --port 8000
curl -H "X-API-Key: $XSCIENTIST_API_KEY" http://127.0.0.1:8000/health
```

Or embed it:

```python
from xscientist import ServiceSettings, create_app

app = create_app(
    ServiceSettings(
        output_root="./research-output",
        max_workers=2,
        max_output_chars=200_000,
    )
)
```

Endpoints:

- `GET /health`
- `POST /v1/projects`
- `GET /v1/jobs`
- `GET /v1/jobs/{job_id}`
- `GET /v1/jobs/{job_id}/logs`
- `POST /v1/jobs/{job_id}/cancel`
- `POST /v1/jobs/{job_id}/resume`
- `GET /v1/papers`
- `GET /v1/papers/{folder}`
- `GET /v1/shortlist`
- `GET /v1/boards/submission`
- `GET /v1/boards/rewrite`
- `GET /v1/projects/{project}/research/status`
- `GET /v1/projects/{project}/research/audit`
- `GET /v1/projects/{project}/research/dag`

Example request:

```json
{
  "project": "demo",
  "question": "Why does the mechanism fail out of distribution?",
  "autopilot": "balanced",
  "allow_synthetic_data": true,
  "max_project_tokens": 400000,
  "bfts_config": "default"
}
```

The service owns its filesystem boundary:

- `project` must be a single directory name;
- `topic`, `ideas`, `data_dir`, and custom BFTS configs must resolve inside
  `work_dir`; plain-language `question` values contain no host path;
- the request cannot override the configured `output_root` through fields;
- free-form `extra_args` are rejected by the HTTP adapter. Add a typed field
  when the service needs to expose another workflow option.

These restrictions apply to the HTTP adapter only. Trusted local Python SDK and
CLI callers retain their normal path flexibility.

Read-only query examples:

```bash
curl -H "X-API-Key: $XSCIENTIST_API_KEY" \
  "http://127.0.0.1:8000/v1/papers?sort_by=quality&limit=20"

curl -H "X-API-Key: $XSCIENTIST_API_KEY" \
  "http://127.0.0.1:8000/v1/papers/paper-folder-name"

curl -H "X-API-Key: $XSCIENTIST_API_KEY" \
  "http://127.0.0.1:8000/v1/shortlist?target_venue=iclr&require_gate=true&top_n=5"

curl -H "X-API-Key: $XSCIENTIST_API_KEY" \
  "http://127.0.0.1:8000/v1/boards/submission?require_gate=true"

# Metadata-only by default; add include_summaries=true only for authorized users.
curl -H "X-API-Key: $XSCIENTIST_API_KEY" \
  "http://127.0.0.1:8000/v1/projects/demo/research/dag?ref=HEAD"
```

The service always reads these views from its configured `output_root`;
requests cannot select another research directory. Absolute artifact paths
inside that root are returned as relative paths, and absolute paths outside the
root are redacted. A paper-detail `folder` must be one directory name directly
under `<output_root>/papers`; absolute paths, nested paths, traversal, and
symlink escapes are rejected.

The bundled service is intended for local/team integration. It includes API-key
authentication and persistent job metadata. Internet-facing deployments should
also add rate limits, durable external queues/workers, TLS, and centralized
observability. Job stdout/stderr is truncated to `max_output_chars` per stream.

Job metadata is atomically persisted under
`<output-root>/.xscientist/api/jobs` (or `--state-dir`). Completed jobs survive
service restarts; queued/running jobs recovered after a restart are marked
`interrupted` because the original child process cannot be safely reattached.
The logs endpoint returns bounded live tails while a job is running. Cancellation
requests graceful process-group termination before escalation, and resume creates
a new job linked through `resume_of` rather than rewriting the old audit record.

## Packaged resources

The wheel includes runtime assets previously available only from a Git clone:

- default and deep BFTS YAML profiles;
- LaTeX templates;
- review few-shot examples;
- ARA JSON schemas;
- experiment-tree visualization templates.

## API service handoff

For a small internal deployment:

```bash
pip install "xscientist[research,openai,service] @ git+https://github.com/smileformylove/XScientist.git@main"
export XSCIENTIST_API_KEY="replace-with-a-secret"
xscientist serve --host 0.0.0.0 --port 8000 --output-root /srv/xscientist
```

The HTTP layer deliberately submits the heavy workflow through the same
`XScientist` SDK used by Python callers. This keeps provider SDKs and workflow
environment mutations isolated in child processes. For larger deployments,
replace the in-process executor with a durable queue while preserving
`ProjectRequest` and `CommandResult` as the adapter boundary.

Service jobs retain at most `--max-output-chars` characters per output stream
while continuously draining the child process. `--max-workspace-bytes` and
`--max-workspace-files` terminate a job that exceeds its project quota; the
defaults are 10 GiB and 100,000 files.
