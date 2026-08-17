# Privacy and local usage metrics

Usage metrics are disabled by default. Enabling them writes fixed-shape events
to the user's local data directory and never transmits them:

```bash
xscientist metrics status
xscientist metrics enable
xscientist metrics export --json
xscientist metrics disable
```

Each event contains only timestamp, XScientist version, an allow-listed command
category, coarse status, and a coarse duration bucket. The API has no arbitrary
metadata parameter, so research questions, paths, credentials, provider output,
and artifacts cannot be attached accidentally.

For temporary automation, `XSCIENTIST_USAGE_METRICS=1` opts in and
`XSCIENTIST_USAGE_METRICS=0` opts out. `XSCIENTIST_METRICS_DIR` can isolate the
local store in tests or managed workstations.

Metrics are diagnostic counters, not analytics telemetry. Export is for the
user to inspect or share deliberately; XScientist has no upload command.

Run `xscientist privacy audit . --history` before publishing a repository.
