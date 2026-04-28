# XScientist Deep Optimization Notes

This note summarizes the deeper optimization layer in XScientist: stronger experiment planning, better writing structure, novelty assessment, baseline management, and quality-oriented model routing.

It is an implementation and operations summary, not a guarantee that generated papers are correct or submission-ready without human review.

## Goals

- Increase research depth through richer experiments and comparisons.
- Improve writing quality through structured academic writing workflows.
- Increase novelty pressure during ideation and ranking.
- Improve baseline selection and statistical evaluation.
- Use stronger models for reasoning-heavy stages while keeping cheaper models for lighter tasks.

## Enhanced Experiment Configuration

`bfts_config_enhanced.yaml` provides a stronger BFTS configuration for deeper runs.

Typical enhancements include:

- More search iterations in early and creative stages.
- Multi-seed evaluation for more stable metrics.
- Dedicated comparison-study stages.
- Adaptive experiment budgets.
- Uncertainty estimation through bootstrap, MCMC, or ensemble-style methods where supported.
- Statistical tests such as t-test, Wilcoxon, and Mann-Whitney U.

Example:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --bfts-config bfts_config_enhanced.yaml \
  --paper-types normal \
  --target-venue neurips \
  --high-quality-mode
```

## Writing Strategy Layer

`ai_scientist/writing_strategies.py` defines writing strategies for:

- conference papers
- workshop papers
- journal-style papers
- extended abstracts
- technical reports

The writing layer evaluates:

- novelty
- rigor
- depth
- clarity
- impact
- structure

## Innovation Enhancement

`ai_scientist/innovation_enhancer.py` provides:

- novelty evaluation
- feasibility assessment
- impact prediction
- related-paper search hooks
- breakthrough idea generation
- research-depth enhancement

These signals are best used to rank and triage ideas. They should not be treated as final novelty claims.

## Baseline And Evaluation Support

`ai_scientist/baseline_system.py` helps with:

- baseline suggestions
- baseline configuration
- multi-dimensional evaluation
- efficiency and robustness checks
- confidence intervals and significance tests

Generated baseline plans still need human validation, especially for domain-specific benchmarks.

## Model Routing

A practical high-quality run often uses:

- stronger reasoning models for planning and review
- capable coding models for implementation edits
- writing-specialized models for manuscript drafting
- lighter models for citation and plotting support

Prefer explicit provider-prefixed model names when using non-default providers, for example `openai/gpt-4.1` or `zhipu/glm-4-plus`.

## Recommended High-Quality Command

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --paper-types normal \
  --target-venue neurips \
  --auto-adjust-paper-type \
  --submission-mode \
  --rank-ideas \
  --top-k-ideas 2 \
  --high-quality-mode \
  --quality-preset publishable \
  --review-strategy depth \
  --improvement-rounds 3 \
  --bfts-config bfts_config_enhanced.yaml
```

## Quality Checklist

Before treating an output as submission-ready, check:

- the final PDF compiles cleanly
- figures and captions match the claims
- citations are real and relevant
- baseline selection is appropriate
- metrics are reproducible
- review issues and repair tasks are resolved or explicitly waived
- `research_manager.py submission-board` and `repair-board` do not show unresolved blockers

## Related Guides

- [RESEARCH_GENERATOR_README.md](RESEARCH_GENERATOR_README.md)
- [PROFESSIONAL_WRITING_README.md](PROFESSIONAL_WRITING_README.md)
- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md)
