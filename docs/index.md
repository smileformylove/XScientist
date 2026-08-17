# XScientist

XScientist turns a falsifiable question into an inspectable research history:
experiments, failures, evidence, bounded inferences, reviews, claims, and the
next scientific action are connected in one offline DAG.

Start with the provider-free journey. It uses no network or model and costs
nothing:

```bash
python -m pip install "xscientist==0.1.3"
xscientist demo ./retrieval-study --autopilot --open
xscientist status ./retrieval-study
```

The fixture deliberately ends with a contested broad claim. A successful
software run is therefore not misrepresented as successful scientific closure.

## Choose a path

| Goal | Start | Provider needed |
| --- | --- | --- |
| See an evidence DAG | `xscientist demo DIR --autopilot` | No |
| Diagnose a workspace | `xscientist doctor --workspace DIR` | No paid call |
| Run a guarded study | `xscientist start DIR` | Yes |
| Manage a long run | `xscientist runs list --workspace DIR` | No additional call |
| Integrate a producer | `xscientist conformance init KIT` | No |
| Check compatibility | `xscientist upgrade check --workspace DIR` | No; `--online` is explicit |

Read [Getting started](GETTING_STARTED.md) for the practical workflow or the
[Research protocol](RESEARCH_PROTOCOL_V2.md) for the scientific model.

!!! warning
    XScientist is alpha research software. Generated code belongs inside the
    configured isolation boundary. Machine-generated claims stay unverified
    until the required independent evidence and review gates exist.
