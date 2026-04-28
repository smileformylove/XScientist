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

`continuous_paper_generator.py` supports:

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

`research_manager.py` provides:

- batch listing and summaries
- paper listing and search
- output-index rebuilding
- submission, rewrite, repair, process, and evolution boards
- cleanup helpers

Useful commands:

```bash
python3 research_manager.py rebuild-index
python3 research_manager.py submission-board --top 5 --require-gate
python3 research_manager.py rewrite-board --top 10
python3 research_manager.py repair-board --top 20
python3 research_manager.py process-board --status blocked --top 30
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

`continuous_research_daemon.py` and `run_stable_daemon.sh` support:

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
python3 preflight_check.py --strict
python3 validate_repo.py
make smoke
```

`make smoke` runs syntax checks, unit tests, repository validation, and import smoke checks.

## Related Files

- `continuous_paper_generator.py`
- `continuous_research_daemon.py`
- `research_manager.py`
- `run_project.py`
- `ai_scientist/config/paths.py`
- `docs/guides/OUTPUT_DIRECTORIES.md`
- `docs/guides/RESEARCH_GENERATOR_README.md`
