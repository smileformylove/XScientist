# XScientist

XScientist turns your idea into an inspectable research history: the question,
falsifiable expectations, experiments, failures, web sources, video assets,
time-localized evidence, bounded inferences, reviews, claims, and next
scientific action stay connected.

Start with your own idea. This first step uses no network or model and costs
nothing:

```bash
python -m pip install \
  "xscientist==0.1.5"
xscientist explore ./my-study
```

To inspect a complete contested evidence history, run the bundled fixture:

```bash
xscientist demo ./retrieval-study --autopilot --open
xscientist status ./retrieval-study
```

This site documents the published `0.1.5` release.

The fixture deliberately ends with a contested broad claim. A successful
software run is therefore not misrepresented as successful scientific closure.

## Choose a path

| Goal | Start | Provider needed |
| --- | --- | --- |
| Turn your idea into a testable plan | `xscientist explore DIR` | No |
| See an evidence DAG | `xscientist demo DIR --autopilot` | No |
| Search web or video sources | `xscientist research web search QUERY --vertical video` | Provider key may be needed |
| Inspect a local video asset | `xscientist research video add FILE` | FFmpeg for import/extraction |
| Audit trace/replay/verification | `xscientist audit DIR` | No |
| Save or safely reverse history | `xscientist history` | No |
| Diagnose a workspace | `xscientist doctor --workspace DIR` | No paid call |
| Run a guarded study | `xscientist start DIR` | Yes |
| Manage a long run | `xscientist runs list --workspace DIR` | No additional call |
| Integrate a producer | `xscientist conformance init KIT` | No |
| Check compatibility | `xscientist upgrade check --workspace DIR` | No; `--online` is explicit |

Read [Getting started](GETTING_STARTED.md) for the practical workflow or the
[Research protocol](RESEARCH_PROTOCOL_V2.md) for the scientific model.

For source discovery and multimodal work, see [Auditable Web Search](WEB_SEARCH.md)
and [Video Assets and Evidence](VIDEO_EVIDENCE.md). Search candidates remain
discovery leads until selected; video observations retain source hashes and
timestamps but do not bypass evidence or review gates.

For a frank current-state review, including usability, provider provenance,
scientific-quality limits, exploration coverage, clue transparency, and
feedback/self-evolution gaps, see [the 2026 project audit](PROJECT_AUDIT_2026-08.md).

!!! warning
    XScientist is alpha research software. Generated code belongs inside the
    configured isolation boundary. Machine-generated claims stay unverified
    until the required independent evidence and review gates exist.
