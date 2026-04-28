# XScientist Auto-Improvement Guide

XScientist includes a review-driven improvement loop for generated papers. The loop reads review output, extracts actionable issues, applies targeted manuscript changes, reruns checks, and records improvement artifacts.

## Main Components

| Component | Purpose |
| --- | --- |
| `ai_scientist/perform_auto_improvement.py` | Review-driven paper improvement |
| `ai_scientist/review_strategies.py` | Review presets and iteration control |
| `ai_scientist/improvement_reporter.py` | JSON, Markdown, and text improvement reports |
| `continuous_paper_generator.py` | Batch integration for improvement rounds |
| `run_project.py` | Project-level integration for improvement rounds |

## Review Strategies

Supported strategy names include:

- `standard`: balanced review
- `fast`: fast pass for obvious issues
- `depth`: deeper review and repair planning
- `neurips`, `iclr`, `cvpr`, `journal`, `nature`: venue-oriented presets

Example:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --paper-types normal \
  --review-strategy depth \
  --improvement-rounds 3
```

## CLI Usage

Disable improvement and only perform review:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --improvement-rounds 0
```

Run one improvement round:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --improvement-rounds 1
```

Run a stronger submission-oriented pass:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --paper-types normal \
  --target-venue neurips \
  --auto-adjust-paper-type \
  --submission-mode \
  --high-quality-mode \
  --quality-preset publishable \
  --review-strategy depth \
  --improvement-rounds 4
```

## Output Artifacts

A paper directory may contain:

```text
paper_YYYYMMDD_HHMMSS_my_idea_normal/
├── idea.json
├── idea.md
├── experiment/
├── latex/
│   ├── template.tex
│   ├── template_backup_round1.tex
│   └── ...
├── paper.pdf
├── paper_initial.pdf
├── reviews/
│   ├── initial/
│   ├── round_1/
│   ├── round_2/
│   └── final/
├── improvement_record.json
└── improvement_reports/
    ├── improvement_report_YYYYMMDD_HHMMSS.json
    ├── improvement_report_YYYYMMDD_HHMMSS.md
    └── improvement_chart_YYYYMMDD_HHMMSS.txt
```

Newer review workflows may also persist structured issue ledgers, repair plans, TODO closure snapshots, and self-evolution artifacts.

## Python API

```python
from ai_scientist.perform_auto_improvement import improve_paper_with_review

result = improve_paper_with_review(
    paper_dir="/path/to/paper",
    text_review=text_review,
    img_review=img_review,
    max_rounds=3,
    model="glm-4-plus",
)

print(result["rounds_completed"])
```

Use strategy helpers directly:

```python
from ai_scientist.review_strategies import ReviewStrategyManager

strategy = ReviewStrategyManager.recommend_strategy(
    paper_type="normal",
    time_constraint="normal",
    quality_requirement="high",
)
```

## Troubleshooting

Low improvement after several rounds:

- Increase `--review-strategy` depth.
- Use a stronger `--model-review` or `--model-writeup`.
- Check whether the initial paper already passed the relevant gate.

LaTeX compilation fails after a rewrite:

- Inspect `latex/template_backup_round*.tex`.
- Check the generated review and repair artifacts for malformed edits.
- Increase `--writeup-retries` for transient compile failures.

Improvement stops too early:

- Lower the minimum improvement threshold where available.
- Increase `--improvement-rounds`.
- For high-quality workflows, inspect blocker counts and quality-gate results.
