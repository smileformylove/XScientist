# XScientist Project Runner

`xscientist project` runs a single research project end to end: ideation, experiment execution, writeup, review, repair, and artifact packaging. `run_project.py` remains a compatibility alias for source checkouts and older automation.

Use it when you want a focused project directory rather than a long-running daemon or a batch of independent papers.

## Quick Start

Login is required for guarded entrypoints:

```bash
xscientist auth login --user <your_name>
xscientist auth status
```

For a first complete run, provide the research question directly. The preset
checks the isolated execution runtime before any research-model call, explores
and ranks alternatives, resumes interrupted BFTS work, runs quality gates,
synthesizes evidence-bound candidate insights, records Research VCS history,
and exports the unified scientific DAG:

```bash
xscientist doctor --deep --task research
xscientist project my_research \
  --question "Why does retrieval-guided reflection fail out of distribution?" \
  --autopilot discovery
```

`--autopilot` without a value selects the cost-bounded `balanced` profile.
Choose `discovery` for stronger branch/rival-hypothesis exploration, or
`publication` for a multi-agent review board and submission gates. Every profile
has finite BFTS budgets and stopping conditions. XScientist derives
`00_config/autopilot_bfts.yaml` without modifying your source config, enforces
isolation and no experiment network, and caps total BFTS tokens/wall time at
800k/4h (`balanced`), 1.5m/8h (`discovery`), or 1.2m/8h (`publication`). Lower
user-supplied limits are preserved.

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
    ├── autopilot_run.json
    ├── progress.json
    ├── insight_report.json
    └── insight_report.md
```

The offline evidence browser is exported outside the Research VCS working tree
to `<output_root>/views/<project>/research-dag/research-dag.html`, so generating a
view never dirties scientific history.

## Common Options

| Option | Purpose | Default |
| --- | --- | --- |
| `project_dir` | Project directory name or absolute path | required |
| `--output-root` | Output root for relative project names | resolved output root |
| `--topic` | Topic markdown file | none |
| `--question` | Plain-language research question; creates the topic artifact | none |
| `--ideas` | Existing idea JSON file | none |
| `--autopilot [profile]` | `balanced`, `discovery`, or `publication` end-to-end preset | disabled |
| `--resume` | Reuse successful results and valid BFTS checkpoints | disabled; enabled by autopilot |
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

Resume after interruption (the original question and generated ideas are reused):

```bash
xscientist project my_research \
  --autopilot discovery
```

## Scientific interpretation boundary

Autopilot is operationally end to end, but it does not collapse scientific
roles. Internal agents may propose, execute, criticize, repair, and synthesize.
They do not count as independent replication. `insight_report.json` therefore:

- allows only low/medium confidence and exact run evidence selectors;
- retains rival hypotheses, uncertainty, and the next high-information experiment;
- labels every insight `machine_synthesized_unverified`;
- projects those insights into the Research VCS/DAG as draft claims behind a
  hold gate until independent verification exists.

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
