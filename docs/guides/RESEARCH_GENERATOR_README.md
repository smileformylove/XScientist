# XScientist Continuous Paper Generator

`xscientist batch` creates batches of papers from a topic or an existing ideas file. It supports multiple paper types, parallel workers, review/repair rounds, high-quality gates, and batch-level progress reports. `continuous_paper_generator.py` remains a compatibility alias.

Use it when you want more throughput than `run_project.py` but do not need a persistent daemon loop.

## Paper Types

The CLI accepts these paper types:

| Type | Typical use | Template family |
| --- | --- | --- |
| `icbinb` | short workshop paper | ICBINB / ICLR-style workshop |
| `normal` | standard conference paper | ICML-style |
| `journal` | longer journal-style paper | ICML-style |
| `extended` | extended abstract | ICBINB-style |

Venue intent is configured separately with `--target-venue`. For example, use `--paper-types normal --target-venue neurips --auto-adjust-paper-type` for a NeurIPS-oriented standard paper.

## Output Layout

Runtime artifacts are written under the active XScientist output root:

```text
<output_root>/
├── cache/
├── ideas/
├── experiments/
├── papers/
│   └── paper_YYYYMMDD_HHMMSS_idea_name_type/
│       ├── idea.json
│       ├── idea.md
│       ├── experiment/
│       ├── latex/
│       ├── paper.pdf
│       ├── reviews/
│       └── logs/
└── batches/
    └── batch_YYYYMMDD_HHMMSS/
        ├── ideas/
        ├── progress.json
        └── final_report.json
```

The output root follows:

```text
RESEARCH_OUTPUT_DIR > AI_SCIENTIST_OUTPUT_DIR > sibling <repo-name>_outputs
```

## Quick Start

Login is required:

```bash
xscientist auth login --user <your_name>
xscientist auth status
```

Set the model credentials you need. The default models use Zhipu credentials:

```bash
export ZHIPU_API_KEY="your_api_key_here"
```

Generate workshop papers from a topic:

```bash
xscientist batch \
  --topic topic.md \
  --num-ideas 5 \
  --paper-types icbinb
```

Generate all supported paper types:

```bash
xscientist batch \
  --topic topic.md \
  --num-ideas 3 \
  --all-types
```

Generate from existing ideas:

```bash
xscientist batch \
  --ideas existing_ideas.json \
  --paper-types normal journal
```

Use parallel workers:

```bash
xscientist batch \
  --topic topic.md \
  --all-types \
  --num-workers 2
```

Process selected idea indices:

```bash
xscientist batch \
  --ideas my_ideas.json \
  --all-types \
  --idea-indices 0,1,2
```

## Quality-Oriented Runs

```bash
xscientist batch \
  --topic topic.md \
  --paper-types normal \
  --target-venue neurips \
  --auto-adjust-paper-type \
  --submission-mode \
  --rank-ideas \
  --top-k-ideas 2 \
  --high-quality-mode \
  --quality-preset publishable \
  --review-strategy depth \
  --improvement-rounds 3
```

## Batch Management

List batches:

```bash
xscientist manager list-batches
```

Inspect a batch:

```bash
xscientist manager batch-summary 20240101_120000
```

List papers:

```bash
xscientist manager list-papers
xscientist manager list-papers --type normal
xscientist manager list-papers --detailed
```

Search generated papers:

```bash
xscientist manager search-papers "transformer"
```

Clean old artifacts:

```bash
xscientist manager cleanup --days 30 --dry-run
xscientist manager cleanup --days 30
```

## Useful Options

| Option | Purpose |
| --- | --- |
| `--research-dir` | Explicit output root |
| `--batch-name` | Stable batch name instead of timestamp |
| `--topic` | Topic markdown file |
| `--ideas` | Existing idea JSON file |
| `--num-ideas` | Number of generated ideas |
| `--paper-types` | One or more supported paper types |
| `--all-types` | Generate every supported paper type |
| `--num-workers` | Parallel worker count |
| `--idea-indices` | Comma-separated idea indices |
| `--rank-ideas` | Rank generated or loaded ideas |
| `--top-k-ideas` | Keep only top-ranked ideas |
| `--target-venue` | Venue intent: `neurips`, `iclr`, `cvpr`, `journal`, or `nature` |
| `--auto-adjust-paper-type` | Adjust paper type to fit target venue |
| `--high-quality-mode` | Enable stronger quality workflow |
| `--quality-preset` | `balanced`, `high`, or `publishable` |
| `--override-strict-fallbacks` | Continue despite strict fallback events |

## Troubleshooting

PDF generation fails:

```bash
which pdflatex
xscientist batch --topic topic.md --writeup-retries 5
```

Memory pressure is high:

```bash
xscientist batch --topic topic.md --num-workers 1
```

Provider rate limits are tight:

```bash
xscientist batch \
  --topic topic.md \
  --model-writeup glm-4-air \
  --model-citation glm-4-flash
```

## Related Files

- `continuous_paper_generator.py`: batch generation entrypoint
- `research_manager.py`: output index and boards
- `ai_scientist/config/paths.py`: output-path and paper-type configuration
- `run_project.py`: focused single-project runner
- `continuous_research_daemon.py`: long-running scheduler
