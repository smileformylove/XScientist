# XScientist Professional Writing Guide

XScientist includes professional writing utilities for venue-aware manuscript structure, section-level drafting, and quality evaluation.

The CLI paper type and the writing template are related but not identical:

- CLI `--paper-types` accepts `icbinb`, `normal`, `journal`, and `extended`.
- Professional templates include venue styles such as `neurips`, `iclr`, and `cvpr`.
- Use `--target-venue` to express venue intent in CLI workflows.

## Main Components

| Component | Purpose |
| --- | --- |
| `ExpertSectionWriter` | Generates outlines, sections, and full manuscripts |
| `ProfessionalPaperEvaluator` | Scores manuscript quality across writing dimensions |
| `recommend_template()` | Chooses a venue-style template from idea metadata |
| `continuous_paper_generator.py` | Integrates professional writing into batch generation |

## CLI Usage

Generate a NeurIPS-oriented standard paper:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --paper-types normal \
  --target-venue neurips \
  --auto-adjust-paper-type \
  --improvement-rounds 3
```

Generate a journal-oriented paper:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --paper-types journal \
  --target-venue journal \
  --high-quality-mode \
  --quality-preset publishable
```

## Python API

Use the batch generator directly:

```python
from continuous_paper_generator import ContinuousPaperGenerator

generator = ContinuousPaperGenerator(
    research_dir="/path/to/my_xscientist_outputs",
)

ideas_json = generator.generate_ideas(
    topic_file="examples/example_topic.md",
    num_ideas=3,
)

with open(ideas_json, "r", encoding="utf-8") as f:
    ideas = json.load(f)

for idea in ideas:
    result = generator.generate_paper_with_professional_writing(
        idea=idea,
        paper_type="normal",
        experiment_results=None,
        model="claude-3-5-sonnet",
        enable_evaluation=True,
        target_venue="neurips",
    )
    print(result["status"], result.get("latex_path"))
```

Use writing components directly:

```python
from ai_scientist.professional_writing_system import (
    ExpertSectionWriter,
    ProfessionalPaperEvaluator,
)

writer = ExpertSectionWriter(template="neurips")
outline = writer.generate_detailed_outline(idea, experiment_results)
paper = writer.write_full_paper(idea, experiment_results, iterative=True)

evaluator = ProfessionalPaperEvaluator(template="neurips")
evaluation = evaluator.evaluate_paper_quality(paper, idea)
```

## Template Summary

| Template | Focus | Typical shape |
| --- | --- | --- |
| `neurips` | theoretical machine learning and clear contribution framing | abstract, introduction, related work, method, experiments, results, discussion, conclusion |
| `iclr` | representation learning and method intuition | abstract, introduction, background, method, experiments, related work, conclusion |
| `cvpr` | visual tasks and qualitative/quantitative evidence | abstract, introduction, related work, method, experiments, conclusion |
| `journal` | deeper discussion and reproducibility | longer introduction, related work, background, method, experiments, discussion, conclusion |

## Quality Dimensions

The evaluator focuses on:

- Structure
- Content coverage
- Innovation
- Rigor
- Clarity
- Professionalism

Use evaluation output as a planning signal for rewrite rounds, not as a substitute for human review.

## Best Practices

- Keep CLI paper type stable (`normal`, `journal`, etc.) and pass venue intent through `--target-venue`.
- Use `--high-quality-mode` for submission-oriented runs.
- Pair professional writing with structured review and repair artifacts.
- Inspect generated LaTeX before treating a PDF as submission-ready.

## Related Guides

- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md)
- [ADAPTIVE_LEARNING_README.md](ADAPTIVE_LEARNING_README.md)
- [RESEARCH_GENERATOR_README.md](RESEARCH_GENERATOR_README.md)
