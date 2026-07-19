# XScientist Project Runner

`xscientist project` runs a single research project end to end: ideation, experiment execution, writeup, review, repair, and artifact packaging. `run_project.py` remains a compatibility alias for source checkouts and older automation.

Use it when you want a focused project directory rather than a long-running daemon or a batch of independent papers.

## Quick Start

Login is required for guarded entrypoints:

```bash
xscientist auth login --user <your_name>
xscientist auth status
```

Run one project from a topic file:

```bash
xscientist project my_research \
  --topic topic.md
```

Run several ideas in parallel:

```bash
xscientist project my_research \
  --topic topic.md \
  --num-ideas 3 \
  --parallel \
  --num-workers 2 \
  --improvement-rounds 2
```

Process existing ideas:

```bash
xscientist project my_research \
  --ideas existing_ideas.json \
  --idea-indices 0,2,4 \
  --parallel
```

## Output Location

Relative `project_dir` values are created under:

```text
<output_root>/projects/<project_dir>/
```

The output root follows the standard XScientist resolution:

```text
RESEARCH_OUTPUT_DIR > AI_SCIENTIST_OUTPUT_DIR > sibling <repo-name>_outputs
```

Use `--output-root` when you want an explicit path for one invocation:

```bash
xscientist project my_research \
  --output-root /path/to/my_xscientist_outputs \
  --topic topic.md
```

## Project Layout

```text
my_research/
├── 01_ideas/
│   └── generated_ideas.json
├── 02_experiments/
│   └── <timestamped_idea_run>/
│       ├── idea.json
│       ├── logs/
│       ├── plots/
│       ├── reviews_round_*/
│       └── *.pdf
├── 03_papers/
│   └── *_final.pdf
└── 04_logs/
    └── progress.json
```

## Common Options

| Option | Purpose | Default |
| --- | --- | --- |
| `project_dir` | Project directory name or absolute path | required |
| `--output-root` | Output root for relative project names | resolved output root |
| `--topic` | Topic markdown file | none |
| `--ideas` | Existing idea JSON file | none |
| `--model-ideation` | Ideation model | `glm-4-flash` |
| `--num-ideas` | Number of generated ideas | `3` |
| `--num-reflections` | Reflection rounds per idea | `5` |
| `--parallel` | Process multiple ideas in parallel | disabled |
| `--num-workers` | Parallel worker count | `2` |
| `--idea-indices` | Comma-separated idea indices | all selected ideas |
| `--rank-ideas` | Rank ideas before selection | disabled |
| `--top-k-ideas` | Limit ranked ideas | none |
| `--submission-mode` | Enable submission-oriented defaults | disabled |
| `--workflow-mode` | Research orchestration mode | `adaptive` |
| `--improvement-rounds` | Review/repair rounds per paper | `1` |
| `--skip-ideation` | Reuse existing ideas | disabled |
| `--skip-experiment` | Skip experiment execution | disabled |
| `--writeup-type` | Paper type: `normal`, `icbinb`, `journal`, or `extended` | `icbinb` |
| `--override-strict-fallbacks` | Continue despite strict fallback events | disabled |

## Example Workflows

Fast single-paper run:

```bash
xscientist project quick_paper \
  --topic topic.md \
  --improvement-rounds 1
```

Higher-quality run:

```bash
xscientist project high_quality \
  --topic topic.md \
  --num-ideas 1 \
  --high-quality-mode \
  --quality-preset publishable \
  --target-venue neurips \
  --auto-adjust-paper-type \
  --num-cite-rounds 25
```

Resume after interruption by skipping ideation and selecting remaining ideas:

```bash
xscientist project my_research \
  --skip-ideation \
  --idea-indices 2,3,4 \
  --parallel
```

## Monitoring

Progress is written to:

```text
<project_dir>/04_logs/progress.json
```

Use `xscientist manager` for repository-wide boards:

```bash
xscientist manager rebuild-index
xscientist manager submission-board --top 5 --require-gate
xscientist manager rewrite-board --top 10
xscientist manager repair-board --top 20
```
