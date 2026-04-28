# XScientist Autonomous Evolution Guide

Autonomous evolution turns review results, agent feedback, repair outcomes, and historical lessons into reusable guidance for later runs.

It is not a claim that the system can replace scientific judgment. It is an engineering loop for preserving and reusing what the system learned from previous failures and repairs.

## Main Capabilities

- Self-reflection over current paper state
- External agent feedback intake
- Multi-source feedback aggregation
- Evolution strategy generation
- Validation of applied improvements
- Knowledge-base updates for future runs

## Basic Usage

```python
from continuous_paper_generator import ContinuousPaperGenerator

generator = ContinuousPaperGenerator(enable_learning=True)

result = await generator.generate_paper_with_evolution(
    idea=idea,
    paper_type="normal",
    evolution_rounds=3,
)

print(result.get("evolution_report"))
```

## Register External Agents

```python
from ai_scientist.agent_interface import ExampleWritingCriticAgent

writing_agent = ExampleWritingCriticAgent()
generator.register_external_agent(writing_agent, group="writing")

result = await generator.generate_paper_with_evolution(
    idea=idea,
    paper_type="normal",
    enable_external_agents=True,
)
```

## Custom Agent Shape

External agents should inherit from `BaseAgent` and return standardized feedback:

```python
from ai_scientist.agent_interface import BaseAgent, AgentCapability, StandardFeedback

class MyCustomAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MyCustomAgent",
            version="1.0",
            capabilities=[AgentCapability.DOMAIN_EXPERTISE],
        )

    async def analyze(self, paper_data, current_state, context=None):
        return StandardFeedback.create(
            score=4.0,
            issues=["The ablation plan is still thin."],
            suggestions=["Add at least two targeted ablations."],
            strengths=["The method section is clear."],
        )
```

## Human Feedback

```python
generator.submit_feedback(
    source="human",
    feedback={
        "score": 3.5,
        "issues": ["Experiments are under-specified."],
        "suggestions": ["Add dataset, metric, and baseline details."],
    },
    metadata={"reviewer": "internal"},
)
```

## Full Workflow

```python
from continuous_paper_generator import ContinuousPaperGenerator
from ai_scientist.agent_interface import (
    ExampleWritingCriticAgent,
    ExampleTechnicalReviewerAgent,
)

generator = ContinuousPaperGenerator(enable_learning=True)
generator.register_external_agent(ExampleWritingCriticAgent(), group="quality")
generator.register_external_agent(ExampleTechnicalReviewerAgent(), group="technical")

result = await generator.generate_paper_with_evolution(
    idea=my_idea,
    paper_type="normal",
    enable_external_agents=True,
    evolution_rounds=3,
)
```

## Monitoring

```python
status = generator.get_evolution_status()
guidance_report = generator.get_guidance_report()

print(status.get("evolution", {}))
print(guidance_report.get("paper_registry", {}))
```

## Outputs

Evolution outputs may include:

- Per-paper evolution reports
- Agent interaction summaries
- Reusable self-evolution playbooks
- Knowledge-base snapshots
- Repair-lane recommendations

## Best Practices

- Start with a small set of high-signal agents.
- Prefer concrete issues and suggestions over broad ratings.
- Review self-evolution artifacts before letting them steer long-running daemon behavior.
- Keep human review in the loop for paper claims, citations, and experimental validity.

## Related Guides

- [ADAPTIVE_LEARNING_README.md](ADAPTIVE_LEARNING_README.md)
- [PAPER_METADATA_README.md](PAPER_METADATA_README.md)
- [AUTO_IMPROVEMENT_README.md](AUTO_IMPROVEMENT_README.md)
