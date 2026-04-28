# XScientist Output Directories

## Overview

XScientist writes runtime artifacts outside the repository by default. This keeps source code, generated papers, experiment logs, model caches, and daemon artifacts cleanly separated.

The source of truth is `ai_scientist/config/paths.py`.

- Default output root: sibling `<repo-name>_outputs`
- Default for this repository: `../XScientist_outputs`
- Environment priority: `RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > default sibling directory
- Fallback: if the repository parent directory is not writable, XScientist uses a system data directory such as `~/.local/share/ai_scientist/research`

## Layout

```text
<output_root>/
├── cache/              # HuggingFace, Torch, wandb, and similar runtime caches
├── ideas/              # Generated or collected idea JSON files
├── experiments/        # Experiment outputs
├── projects/           # Complete run_project.py project directories
├── papers/             # Per-paper directories from continuous_paper_generator.py
└── batches/            # Continuous-generation batch progress and reports
```

## Check The Active Path

```bash
python3 - <<'PY'
from ai_scientist.config.paths import resolve_output_path
print(resolve_output_path())
PY
```

## Override The Output Root

Set `RESEARCH_OUTPUT_DIR` explicitly for reproducible local or server runs:

```bash
export RESEARCH_OUTPUT_DIR="/path/to/my_xscientist_outputs"
python3 run_project.py my_project --topic examples/example_topic.md
```

`AI_SCIENTIST_OUTPUT_DIR` is still supported for compatibility, but new scripts and documentation should prefer `RESEARCH_OUTPUT_DIR`.

## Python API

```python
from ai_scientist.config import paths

output_root = paths.resolve_output_path()
exp_dir = paths.get_experiment_dir("my_idea", attempt_id=0)
idea_path = paths.get_idea_path("my_idea")
project_dir = paths.get_project_dir("my_project")

paths.ensure_output_dirs()
```

## Migrating Older Local Outputs

Older local runs may have written to `_outputs/`, `research_output/`, `experiments/`, or `projects/` inside the repository. Move those artifacts into the current output root after checking for name collisions.

```bash
export RESEARCH_OUTPUT_DIR="/path/to/my_xscientist_outputs"
mkdir -p "$RESEARCH_OUTPUT_DIR"/{cache,ideas,experiments,projects,papers,batches}

mv _outputs/ideas/* "$RESEARCH_OUTPUT_DIR/ideas/" 2>/dev/null || true
mv _outputs/experiments/* "$RESEARCH_OUTPUT_DIR/experiments/" 2>/dev/null || true
mv _outputs/projects/* "$RESEARCH_OUTPUT_DIR/projects/" 2>/dev/null || true
mv research_output/papers/* "$RESEARCH_OUTPUT_DIR/papers/" 2>/dev/null || true
mv experiments/* "$RESEARCH_OUTPUT_DIR/experiments/" 2>/dev/null || true
```

## Validation

```bash
python3 validate_repo.py
```

`validate_repo.py` checks repository structure and key imports. Full runs still require login, model credentials, and any required system dependencies.
