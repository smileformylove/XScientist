# XScientist SDK and API

XScientist exposes one public package, `xscientist`. The historical
`ai_scientist` package and repository-root scripts remain compatibility
implementation details; new integrations should depend on `xscientist` only.

## Installation profiles

```bash
pip install xscientist                    # lightweight SDK/protocol surface
pip install "xscientist[full]"            # complete research runtime
pip install "xscientist[full,service]"    # runtime plus FastAPI/Uvicorn
pip install -e ".[full,service,dev]"      # contributor environment
```

## Package boundary

```text
xscientist/                 Public, versioned integration surface
├── client.py               Python SDK and subprocess isolation
├── models.py               Stable request/result data models
├── cli.py                  Unified `xscientist` command
├── service.py              Optional FastAPI service
└── entrypoints.py          Compatibility workflow dispatch

ai_scientist/               Internal implementation and ARA protocol
├── protocol/               Stable on-disk research protocol
├── resources/              Packaged configs and resource lookup
├── treesearch/             Experiment search engine
└── utils/                  Internal workflow components

run_project.py and peers    Legacy adapters; avoid importing in new apps
```

Public compatibility follows semantic versioning for symbols exported by
`xscientist.__all__`. Internal modules can evolve more quickly.

## Python SDK

```python
from xscientist import ProjectRequest, XScientist

client = XScientist(
    output_root="./research-output",
    env={"OPENAI_API_KEY": "..."},
)

request = ProjectRequest(
    project="demo",
    topic="topic.md",
    workflow_mode="program_driven",
    high_quality_mode=True,
    bfts_config="deep",
)

result = client.run_project(request)
if not result.ok:
    raise RuntimeError(result.stderr)
```

Workflows run in a child process. Large research modules, provider SDK state,
and environment mutations therefore do not leak into an embedding web app.

## Unified CLI

```bash
xscientist info
xscientist project demo --topic topic.md
xscientist batch --help
xscientist daemon --help
xscientist ara --help
xscientist auth status
```

Direct compatibility commands (`xscientist-project`, `xscientist-batch`,
`xscientist-daemon`, and `xscientist-ara`) are also installed.

## HTTP API

```bash
xscientist serve --host 0.0.0.0 --port 8000 --output-root ./research-output
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

Example request:

```json
{
  "project": "demo",
  "topic": "topic.md",
  "workflow_mode": "adaptive",
  "num_ideas": 3,
  "bfts_config": "default"
}
```

The bundled service is intended for local/team integration. Internet-facing
deployments should add authentication, rate limits, persistent job storage,
and a dedicated queue/worker system. Job stdout/stderr is kept in memory and
truncated to `max_output_chars` per stream.

## Packaged resources

The wheel includes runtime assets previously available only from a Git clone:

- default and deep BFTS YAML profiles;
- LaTeX templates;
- review few-shot examples;
- ARA JSON schemas;
- experiment-tree visualization templates.
