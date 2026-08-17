# Reproducible example gallery

Every example below starts from the same shipped deterministic fixture. The
profile changes the intended operator journey, not the evidence outcome. Each
run uses no provider, makes no network request, and records `$0.00` model cost.

| Example | Command | Typical local runtime | Scientific outcome |
| --- | --- | ---: | --- |
| Balanced first run | `xscientist demo ./balanced --autopilot` | Seconds | Broad claim contested |
| Discovery orientation | `xscientist demo ./discovery --autopilot --autopilot-profile discovery` | Seconds | Refutation becomes the next research target |
| Publication orientation | `xscientist demo ./publication --autopilot --autopilot-profile publication` | Seconds | Submission remains blocked by held-out evidence |

## Inspect any example

```bash
xscientist status ./balanced
xscientist research log --repo ./balanced
xscientist research guide --repo ./balanced
xscientist research dag --repo ./balanced --output ./balanced-dag
```

The example includes:

- a falsifiable hypothesis and preregistered direction;
- a failed attempt rather than a cleaned-up success-only history;
- supporting and refuting observations;
- a bounded inference that does not outrun the evidence;
- independent review rejecting the broad transfer claim;
- deterministic Autopilot progress, budget, and insight receipts.

## Reproduce the benchmark report

```bash
xscientist benchmark first-run --json > first-run.json
python tools/benchmark_first_run.py --max-seconds 30 --output first-run.json
```

The report never includes the temporary workspace path. Runtime varies by host,
so published comparisons must report XScientist version, Python version,
operating system, profile, and the threshold used.
