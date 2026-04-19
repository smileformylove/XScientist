# XScientist Architecture

## Overview

XScientist is designed as a **research operating system** that can run continuously, stay observable, and produce handoff-ready artifacts for iterative improvement and collaboration.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     XScientist Research OS                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Ideation   │───▶│ Experiments  │───▶│   Writeup    │      │
│  │   Engine     │    │   Engine     │    │   Engine     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         └────────────────────┼────────────────────┘              │
│                              ▼                                   │
│                    ┌──────────────────┐                          │
│                    │  Self-Review &   │                          │
│                    │  Repair Engine   │                          │
│                    └──────────────────┘                          │
│                              │                                   │
│         ┌────────────────────┼────────────────────┐              │
│         ▼                    ▼                    ▼              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Autonomous  │    │   Adaptive   │    │  Knowledge   │      │
│  │  Evolution   │    │   Learning   │    │     Base     │      │
│  │   Engine     │    │   Engine     │    │              │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                    │                    │              │
│         └────────────────────┴────────────────────┘              │
│                              │                                   │
│                    ┌──────────────────┐                          │
│                    │  Daemon Strategy │                          │
│                    │   & Scheduling   │                          │
│                    └──────────────────┘                          │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Ideation Engine
**Location**: `ai_scientist/perform_ideation_temp_free.py`

Generates and ranks research ideas based on:
- Topic constraints
- Literature review
- Novelty assessment
- Feasibility analysis

**Key Features**:
- Temperature-free idea generation
- Multi-criteria ranking
- Source-aware ideation

### 2. Experiments Engine
**Location**: `ai_scientist/treesearch/`

Executes experiments using:
- Best-first tree search (BFTS)
- Agent-based execution
- Parallel experiment management
- Structured logging and journaling

**Key Features**:
- Timeout protection
- Resource management
- Experiment TODO tracking
- Evidence collection

### 3. Writeup Engine
**Location**: `ai_scientist/perform_writeup.py`, `ai_scientist/perform_icbinb_writeup.py`

Generates research papers with:
- Multiple paper formats (NeurIPS, ICBINB, etc.)
- LaTeX compilation
- Figure and table generation
- Citation management

**Key Features**:
- Professional writing system
- Writing guardrails
- Quality gates
- Metadata tracking

### 4. Self-Review & Repair Engine
**Location**: `ai_scientist/perform_llm_review.py`, `ai_scientist/perform_vlm_review.py`

Multi-round review system:
- LLM-based text review
- VLM-based figure review
- Structured issue generation
- Repair planning and execution

**Key Features**:
- Review strategies (novelty, rigor, clarity, reproducibility)
- Priority-based issue tracking (P0, P1, P2)
- Repair coverage metrics
- Regression detection

### 5. Autonomous Evolution Engine
**Location**: `ai_scientist/autonomous_evolution.py`

Self-improvement system that:
- Collects feedback from multiple sources
- Identifies patterns in successes/failures
- Evolves strategies over time
- Integrates external agent guidance

**Key Features**:
- Multi-source feedback (self, external agents, metrics, peer review)
- Evolution actions (improve writing, adjust strategy, learn patterns)
- Evolution history tracking
- External agent callbacks

### 6. Adaptive Learning Engine
**Location**: `ai_scientist/adaptive_learning_engine.py`

Strategy recommendation system:
- Analyzes historical performance
- Recommends strategies based on similar past projects
- Predicts success probability
- Adapts to feedback

**Key Features**:
- Pattern analysis
- Success factor identification
- Self-evolution playbook integration
- Context-aware recommendations

### 7. Knowledge Base
**Location**: `ai_scientist/self_learning_knowledge_base.py`

Persistent memory system:
- Stores project outcomes
- Tracks patterns and lessons learned
- Maintains self-evolution playbook
- Enables cross-project learning

**Key Features**:
- Structured artifact storage
- Similarity search
- Pattern extraction
- Playbook management

### 8. Daemon Strategy & Scheduling
**Location**: `continuous_research_daemon.py`

Long-running orchestration:
- Source queue management
- Quality-based feedback loops
- Failure protection
- Trend reporting
- Handoff brief generation

**Key Features**:
- Multi-source scheduling
- Auto quality governor
- Evidence strategy feedback
- Dashboard serving
- Graceful degradation

## Data Flow

### Single Project Flow

```
Topic/Sources
    │
    ▼
Ideation (generate & rank ideas)
    │
    ▼
Experiments (execute & collect evidence)
    │
    ▼
Writeup (generate paper draft)
    │
    ▼
Self-Review (identify issues)
    │
    ▼
Repair (fix issues)
    │
    ▼
Quality Gates (check readiness)
    │
    ▼
Artifacts (paper, reviews, metrics)
    │
    ▼
Knowledge Base (store learnings)
```

### Daemon Flow

```
Source Queue
    │
    ▼
Strategy Selection (based on feedback)
    │
    ▼
Project Execution (single project flow)
    │
    ▼
Feedback Collection
    │
    ├─▶ Source Quality Feedback
    ├─▶ Quality Strategy Feedback
    ├─▶ Evidence Strategy Feedback
    └─▶ Evolution Feedback
    │
    ▼
Strategy Adjustment
    │
    └─▶ (loop back to Source Queue)
```

## Key Design Principles

### 1. Observability
Every stage produces structured artifacts (JSON/MD) for:
- Progress tracking
- Post-mortem analysis
- Reproducibility
- Handoff

### 2. Feedback Loops
Multiple feedback mechanisms:
- **Self-review loop**: Paper → Review → Repair → Paper
- **Evolution loop**: Execution → Feedback → Strategy → Execution
- **Quality loop**: Metrics → Governor → Strategy → Metrics

### 3. Graceful Degradation
System continues operating even when:
- Individual experiments fail
- Review rounds don't converge
- External services are unavailable

### 4. Modularity
Each component can be:
- Tested independently
- Replaced with alternatives
- Extended with new capabilities

### 5. Long-Running Stability
Designed for continuous operation:
- Timeout protection
- Resource cleanup
- State persistence
- Failure recovery

## Configuration System

### Environment Variables
- `RESEARCH_OUTPUT_DIR`: Output directory
- `OPENAI_API_KEY`, `ZHIPU_API_KEY`, etc.: API keys
- `S2_API_KEY`: Semantic Scholar API key

### Config Files
- `configs/sources/*.json`: Source queue configurations
- `bfts_config*.yaml`: Experiment engine configurations
- `.env`: Local environment overrides

### Runtime Flags
- `--submission-mode`: Stricter quality gates
- `--enable-rewrite-followup`: Auto rewrite after review
- `--auto-*-feedback`: Enable various feedback loops

## Output Structure

```
$RESEARCH_OUTPUT_DIR/
├── projects/              # Individual research projects
│   └── <project_name>/
│       ├── experiment/    # Experiment outputs
│       ├── paper/         # Paper drafts
│       ├── reviews/       # Review artifacts
│       └── manifest.json  # Project metadata
├── ideas/                 # Generated ideas
├── reports/               # Daemon reports
│   ├── trends/           # Trend analysis
│   └── handoff/          # Handoff briefs
└── knowledge_base/        # Cross-project memory
    ├── evolution/        # Evolution history
    └── playbook/         # Self-evolution playbook
```

## Extension Points

### Adding New Paper Types
1. Create writeup function in `ai_scientist/`
2. Register in paper type mapping
3. Add LaTeX template if needed

### Adding New Review Strategies
1. Extend `ai_scientist/review_strategies.py`
2. Add strategy to review engine
3. Update repair planner

### Adding New Feedback Sources
1. Extend `FeedbackSource` enum in `autonomous_evolution.py`
2. Implement feedback collector
3. Register callback

### Adding New Evolution Actions
1. Extend `EvolutionAction` enum
2. Implement action handler
3. Add to evolution engine

## Performance Considerations

### Token Usage
- Prompt caching for repeated contexts
- Token tracking per component
- Budget management

### Parallelization
- Parallel experiment execution
- Concurrent review rounds
- Async feedback collection

### Resource Management
- Timeout enforcement
- Memory cleanup
- Disk space monitoring

## Security & Privacy

### API Key Management
- Never commit `.env` files
- Use environment variables
- Rotate keys regularly

### Output Isolation
- Default output outside repo
- Configurable output paths
- Sensitive data filtering

### Login Guard
- Session-based authentication
- User tracking for audit
- Preflight checks

## Testing Strategy

### Unit Tests
- Component-level tests
- Mock external dependencies
- Fast feedback

### Integration Tests
- End-to-end flows
- Real API calls (optional)
- Artifact validation

### Smoke Tests
- Syntax validation
- Import checks
- Config validation

### Regression Tests
- Known failure modes
- Performance benchmarks
- Output consistency

## Monitoring & Debugging

### Logging
- Structured logging throughout
- Log levels (DEBUG, INFO, WARNING, ERROR)
- Contextual information

### Metrics
- Token usage tracking
- Success/failure rates
- Quality scores
- Timing information

### Debugging Tools
- `research_manager.py`: Index and boards
- `preflight_check.py`: System validation
- `validate_repo.py`: Repository health

## Future Directions

### Planned Enhancements
- Stronger TODO closure integration
- Enhanced evidence-to-figure binding
- Submission-ready consistency checks
- Direct playbook-to-rewrite wiring
- Comprehensive English documentation

### Research Opportunities
- Multi-agent collaboration
- Human-in-the-loop refinement
- Cross-domain transfer learning
- Automated experiment design
- Meta-learning for research strategies
