# XScientist Adaptive Learning Guide

XScientist can store prior generation outcomes and use them to recommend writing, review, and improvement strategies for later runs.

Adaptive learning is useful for long-running usage where repeated papers, reviews, failures, and repairs should become reusable operational knowledge.

## Main Components

| Component | Purpose |
| --- | --- |
| `SelfLearningKnowledgeBase` | Stores success/failure patterns, writing insights, review insights, and improvement strategies |
| `PatternAnalyzer` | Analyzes historical success and failure patterns |
| `AdaptiveLearningEngine` | Recommends strategies from the knowledge base |
| `AdaptiveWriter` | Applies learned patterns during section writing |
| `ContinuousPaperGenerator(enable_learning=True)` | Integrates learning, evolution, and agent guidance |

## Basic API

```python
from ai_scientist.self_learning_knowledge_base import SelfLearningKnowledgeBase

kb = SelfLearningKnowledgeBase(research_dir="/path/to/my_xscientist_outputs")

kb.store_paper_experience(
    paper_data=paper_data,
    outcome="accepted",
    final_scores={"structure": 4.5, "content": 4.0},
    reviews=reviews,
    improvements=improvements,
)

similar = kb.find_similar_papers(
    current_idea=idea,
    paper_type="normal",
    top_k=5,
)
```

Strategy recommendation:

```python
from ai_scientist.adaptive_learning_engine import AdaptiveLearningEngine

engine = AdaptiveLearningEngine(kb)

recommendation = engine.recommend_strategy(
    idea=idea,
    paper_type="normal",
    context={"target_venue": "neurips"},
)

print(recommendation["success_probability"])
print(recommendation["self_evolution_guidance"])
```

## Generator Integration

```python
from continuous_paper_generator import ContinuousPaperGenerator

generator = ContinuousPaperGenerator(
    research_dir="/path/to/my_xscientist_outputs",
    enable_learning=True,
)

result = generator.generate_paper_with_adaptive_learning(
    idea=idea,
    paper_type="normal",
    target_venue="neurips",
    experiment_results=results,
    enable_evaluation=True,
    learn_from_result=True,
)

generator.show_learning_status()
```

The CLI batch generator enables learning by default:

```bash
python3 continuous_paper_generator.py \
  --topic examples/example_topic.md \
  --paper-types normal \
  --target-venue neurips \
  --auto-adjust-paper-type
```

## Knowledge-Base Layout

```text
<output_root>/knowledge_base/
├── success_patterns.json
├── failure_patterns.json
├── improvement_strategies.json
├── review_insights.json
├── writing_insights.json
├── self_evolution_history.jsonl
└── self_evolution_playbook.json
```

## Learning Signals

XScientist tracks:

- Successful and failed paper outcomes
- Similarity between ideas and prior papers
- Review issues and repair effectiveness
- Strategy success rate and average improvement
- Score thresholds by dimension
- Self-evolution guidance extracted from repair loops

## Operational Advice

- Expect recommendations to be weak until enough runs have accumulated.
- Store human-verified outcomes when possible.
- Periodically inspect `knowledge_base/` and remove obsolete local experiments if they no longer represent current behavior.
- Treat adaptive recommendations as planning aids, not as final quality judgments.

## Related Guides

- [PROFESSIONAL_WRITING_README.md](PROFESSIONAL_WRITING_README.md)
- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md)
- [AUTONOMOUS_EVOLUTION_README.md](AUTONOMOUS_EVOLUTION_README.md)
