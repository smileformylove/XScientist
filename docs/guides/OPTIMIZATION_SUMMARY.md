# XScientist Optimization Summary

This document summarizes the major operational improvements in XScientist.

## Output Isolation

Runtime outputs are centralized under the active output root:

```text
RESEARCH_OUTPUT_DIR > AI_SCIENTIST_OUTPUT_DIR > sibling <repo-name>_outputs
```

For this repository, the default sibling path is `../XScientist_outputs`.

The output root contains:

```text
<output_root>/
├── cache/
├── ideas/
├── experiments/
├── projects/
├── papers/
└── batches/
```

## Continuous Paper Generation

`python -m xscientist batch` supports:

- topic-based ideation
- existing ideas files
- multiple paper types
- parallel workers
- review and improvement rounds
- batch progress tracking
- final batch reports

## Paper Types

Supported CLI paper types:

- `icbinb`
- `normal`
- `journal`
- `extended`

Use `--target-venue` for venue intent and `--auto-adjust-paper-type` when you want XScientist to align paper type with the venue.

## Research Management

`python -m xscientist manager` provides:

- batch listing and summaries
- paper listing and search
- output-index rebuilding
- submission, rewrite, repair, process, and evolution boards
- cleanup helpers

Useful commands:

```bash
python -m xscientist manager rebuild-index
python -m xscientist manager submission-board --top 5 --require-gate
python -m xscientist manager rewrite-board --top 10
python -m xscientist manager repair-board --top 20
python -m xscientist manager process-board --status blocked --top 30
```

## Guardrails

XScientist now includes:

- login guard for user-facing entrypoints
- preflight checks
- repository validation
- schema-validated daemon and source configs
- strict fallback policy for quality-sensitive modes
- output-root isolation

## Long-Running Operation

`python -m xscientist daemon` and `run_stable_daemon.sh` support:

- source queues
- day/night profiles
- failure backoff
- rewrite follow-up
- source quality feedback
- evidence strategy feedback
- dashboard and operator reports
- handoff reports

## Verification

Recommended local checks:

```bash
python -m xscientist preflight --strict
python -m xscientist validate --full-import-smoke
make smoke
```

`make smoke` runs syntax checks, unit tests, repository validation, and import smoke checks.

## Related Files

- `ai_scientist/apps/batch.py`
- `ai_scientist/apps/daemon.py`
- `ai_scientist/apps/manager.py`
- `ai_scientist/apps/project.py`
- `ai_scientist/config/paths.py`
- `docs/guides/OUTPUT_DIRECTORIES.md`
- `docs/guides/RESEARCH_GENERATOR_README.md`
