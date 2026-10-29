# Research Action Ledger

XScientist records agent operations as immutable `research_action` objects. The
ledger is the durable boundary between policy and execution: a policy chooses
an action ID, while a worker appends receipts for start, interruption, retry,
recovery, and terminal completion.

The receipt stores an input hash and output hash, attempt/sequence numbers,
optional budget values, and Research Object references. It never stores the raw
prompt, tool response, media bytes, or local path. `consumes`, `produces`, and
`supersedes` graph edges make the action auditable alongside video assets,
timed-text indexes, observations, evidence, and experiment results.

## Python API

```python
from xscientist import ResearchActionLedger, ResearchRepository

repo = ResearchRepository("./study")
actions = ResearchActionLedger(repo)
started = actions.begin(
    "video-inspection-42",
    "video.inspect",
    {"asset": "rso-SOURCE_ID", "interval": [12.0, 18.0]},
    consumes=["rso-SOURCE_ID"],
    budget_seconds=90,
)

actions.finish(
    "video-inspection-42",
    "completed",
    {"evidence_id": "rso-EVIDENCE_ID"},
    produces=["rso-EVIDENCE_ID"],
    consumed_seconds=11.4,
)
```

The same input and logical ID return the existing receipt, which makes retries
safe. An input change under an existing ID is rejected. An interrupted action
must be resumed explicitly with `retry=True` or `recovery=True`; the new
attempt links back to the interrupted receipt.

## CLI

```bash
xscientist research action begin ACTION_ID ACTION_TYPE \
  --input-hash sha256:... --consume rso-SOURCE_ID --budget 90 --repo ./study
xscientist research action finish ACTION_ID --status completed \
  --output-hash sha256:... --produce rso-EVIDENCE_ID --repo ./study
xscientist research action list ACTION_ID --repo ./study
```

Action objects are ordinary Research VCS objects. They therefore participate in
privacy scanning, content-addressed identity, Git checkpoints, DAG inspection,
and repository fsck without requiring a separate action database.
