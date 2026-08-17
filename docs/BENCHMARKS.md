# First-run benchmark

The public benchmark measures a usability property: can a clean local process
go from an empty directory to inspectable, scientifically honest status without
credentials, network access, or model cost?

```bash
xscientist benchmark first-run --json
xscientist benchmark first-run --max-seconds 30
```

It creates the deterministic Autopilot fixture, builds the Research VCS DAG,
reads status, records duration and structural counts, and deletes its temporary
workspace. `--workspace DIR` retains a named workspace for inspection.

This is not a model-quality benchmark. It must not be used to claim scientific
performance, autonomous discovery quality, or provider speed. Those require
separate registered datasets, budgets, evidence, and evaluation authority.

CI may use a generous threshold to detect severe first-run regressions. Local
runtime is descriptive and should not be compared across unreported hardware.
