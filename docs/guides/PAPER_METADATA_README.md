# XScientist Paper Metadata Guide

Paper metadata files help other tools, operators, and external agents understand a generated paper's status, quality, and needed assistance.

## Standard Files

A paper directory may contain:

```text
paper_YYYYMMDD_HHMMSS_idea_name_normal/
├── .paper_metadata.json
├── .status.json
├── .agent_comments.json
├── README.md
├── .agent_instructions.md
└── .status_badge.txt
```

## Status Model

Common paper statuses:

- `ideation`
- `generating`
- `draft`
- `under_review`
- `improving`
- `completed`
- `published`

Common quality levels:

- `poor`
- `fair`
- `good`
- `excellent`
- `outstanding`

Common guidance priorities:

- `critical`
- `high`
- `medium`
- `low`
- `optional`

## Creating Metadata During Generation

```python
from continuous_paper_generator import ContinuousPaperGenerator

generator = ContinuousPaperGenerator(enable_learning=True)

result = generator.generate_paper_with_professional_writing(
    idea=idea,
    paper_type="normal",
    target_venue="neurips",
)

metadata = generator.create_paper_metadata_markers(
    paper_dir=result["paper_dir"],
    idea=idea,
    paper_type="normal",
)
```

## Agent Discovery

```python
from ai_scientist.agent_guidance_coordinator import AgentGuidanceAPI

api = AgentGuidanceAPI()

papers = api.discover_papers(
    agent_name="WritingCriticAgent",
    agent_capabilities=["writing_critique", "style_guidance"],
    max_papers=5,
)

for paper in papers:
    print(paper["paper_id"], paper["priority"])
```

## Submitting Guidance

```python
result = api.submit_guidance(
    agent_name="WritingCriticAgent",
    paper_id="paper_20240223_120000_my_idea_normal",
    comment="The introduction should state the contribution more directly.",
    score=3.5,
    issues=["Introduction is too long", "Contribution statement is vague"],
    suggestions=["Shorten the introduction", "Add three explicit contribution bullets"],
    priority="high",
)
```

## Inspecting Paper Metadata

```python
info = api.get_paper_info("paper_20240223_120000_my_idea_normal")

print(info["paper_info"]["title"])
print(info["status"]["current_status"])
print(info["status"]["quality_level"])
print(info["status"]["overall_score"])
```

## Manual Metadata Creation

```python
from ai_scientist.paper_metadata import (
    PaperStatus,
    create_paper_metadata,
    create_standardized_markers,
)

paper_dir = "/path/to/my_xscientist_outputs/papers/paper_20240223_120000_idea_normal"

metadata = create_paper_metadata(
    paper_dir=paper_dir,
    idea={
        "Name": "my_idea",
        "Title": "My Paper",
        "Abstract": "Abstract...",
        "Field": "ML",
        "Task": "Classification",
    },
    paper_type="normal",
)

create_standardized_markers(paper_dir)
metadata.set_status(PaperStatus.DRAFT, "Initial draft complete")
metadata.set_quality(3.8)
```

## Example Metadata

```json
{
  "paper_id": "paper_20240223_120000_my_idea_normal",
  "created_at": "2024-02-23T12:00:00",
  "updated_at": "2024-02-23T12:30:00",
  "paper_type": "normal",
  "target_venue": "neurips",
  "idea_name": "my_idea",
  "title": "My Paper Title",
  "abstract": "Paper abstract...",
  "authors": ["XScientist"],
  "keywords": ["machine learning", "optimization"],
  "field": "Computer Vision",
  "task": "Image Classification"
}
```

## Best Practices

- Create metadata as soon as a paper directory exists.
- Keep status transitions explicit.
- Prefer concrete agent comments with issues and suggestions.
- Use metadata and repair queues to drive `research_manager.py` boards.

## Related Files

- `ai_scientist/paper_metadata.py`
- `ai_scientist/agent_guidance_coordinator.py`
- `ai_scientist/agent_interface.py`
- `research_manager.py`
