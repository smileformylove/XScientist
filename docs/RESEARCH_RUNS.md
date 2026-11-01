# Research run index

XScientist keeps the execution protocols for actions, rollouts, experiments,
and video inspection separate. The run index is the small coordination layer
that makes one run recoverable across those protocols. It records an immutable
sequence of lifecycle steps:

`plan -> delegate -> execute/inspect -> repair -> judge -> stop`

The index is metadata-only. It stores canonical SHA-256 hashes, action object
IDs, and context object IDs. Prompts, model responses, media bytes, local
paths, and credentials stay in their existing stores and are never copied into
a `research_run` object.

## Record a run

Use the Python API when an executor already has structured values:

```python
from xscientist import ResearchRepository, ResearchRunLedger

repo = ResearchRepository("./study")
run = ResearchRunLedger(repo)
run.append(
    "video-followup-01",
    "plan",
    "plan",
    "completed",
    {"question": "private input"},
    output_value={"plan": "bounded temporal inspection"},
    context_refs=["rso-CONTEXT_ID"],
    next_action="video.inspect",
)
```

The CLI accepts hashes directly, so private values do not cross the command
boundary:

```bash
xscientist research run step video-followup-01 inspect-01 \
  --type inspect --status waiting \
  --input-hash sha256:... --context-ref rso-EVIDENCE_ID \
  --requires-human --next-action video.inspect --repo ./study
```

`action_ref` can point to an existing `research_action` receipt. Context refs
are written as `consumes` edges with the `context_only` role: they provide
evidence to the run, but do not grant experiment permission or promote a
claim. A sequence must advance by one; repeating the same step is idempotent,
while a changed input or a skipped sequence is rejected.

## Recover and inspect

List runs or inspect one run with attention first:

```bash
xscientist research run list --repo ./study
xscientist research run show video-followup-01 --repo ./study --json
```

`show` reports the current step, status, context/action references, and any
`waiting` or `blocked` step that requires a person. The summary includes a
`summary_hash` and `raw_content_included: false`, so it can be safely attached
to a handoff or dashboard. The run index does not replace experiment admission,
evidence verification, or rollout audits; those gates remain authoritative.

The lifecycle shape follows the useful part of DeepSeek's
[dsh-superagent-prototype](https://github.com/deepseek-ai/dsh-superagent-prototype):
explicit work states, resumable steps, structured attention, and context that
is separate from authorization. XScientist keeps the implementation local and
content-addressed so existing video assets, evidence packs, and prior
experiments remain reusable without exposing user data.
