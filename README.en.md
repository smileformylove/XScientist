# XScientist

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Chinese README: [README.md](README.md)

> A sustainable, self-improving autonomous research system: idea generation, experiment execution, paper writing, self-review loops, strategy scheduling, and long-running daemon ops.

XScientist is not built to "generate one paper once". It is designed as an operational research pipeline that can run continuously, stay observable, and produce handoff-ready artifacts (plans, evidence, reviews, repair tasks, quality gates, and reports) for iterative improvement and collaboration.

Important notes:

- Cost: running the system calls LLMs / retrieval services and may incur API fees and long runtimes.
- Reliability: model outputs may contain errors or hallucinations; verify key claims, data, and citations yourself.
- Output isolation: by default, run outputs are written outside this git repo (to avoid polluting an open-source repository).

---

## Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs & Observability](#outputs--observability)
- [Example Papers](#example-papers)
- [Docs](#docs)
- [Development](#development)
- [Roadmap](#roadmap)
- [System Architecture](#system-architecture)
- [Contributing & Community](#contributing--community)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation and References](#citation-and-references)

---

## Overview

Think of XScientist as a "research operating system":

- Input: topics / sources / constraints (budget, stop conditions, quality gates)
- Process: ideation -> experiments -> writing -> self-review -> repair/rewrite -> packaging
- Output: reusable research assets (reports, paper drafts, review/repair queues, run index, handoff briefs)

Core loop (simplified):

```mermaid
flowchart LR
  A["Topic / Sources"] --> B["Ideation & Ranking"]
  B --> C["Experiments"]
  C --> D["Writeup & Quality Gates"]
  D --> E["Self-Review & Repair"]
  E --> F["Artifacts + Index + Dossier"]
  F --> G["Daemon Strategy Feedback"]
  G --> B
```

---

## Key Features

- Self-review loop: multi-round self-review produces structured issues + repair plans, and enforces regression/coverage gates.
- Measurable experiment TODO closure: turns "missing evidence" into explicit TODOs and tracks closure progress.
- Long-running daemon: continuous execution, failure protection, source scheduling, trend reports, handoff briefs, and strategy feedback.
- Enhanced feedback system: multi-source feedback collection, real-time health monitoring, trend analysis, automated action generation.
- Observability and replay: critical stage artifacts are written as structured files (JSON/MD) for comparison and post-mortems.
- Engineering safeguards: login guard, preflight/repo validation, config schemas, output directory isolation.

Entrypoints:

- `run_project.py`: single-project end-to-end run (good for local debugging and reproducing a run)
- `continuous_paper_generator.py`: continuous/batch generation
- `continuous_research_daemon.py`: long-running autonomous scheduling
- `research_manager.py`: index + boards (filtering, exporting, packaging)

---

## Quick Start

### 0) Prerequisites

- Python: 3.10+ (3.11 recommended)
- System deps (recommended):
  - LaTeX toolchain (to compile paper PDFs, e.g., TeX Live / MacTeX)
  - `poppler` (PDF processing/extraction)
  - `chktex` (optional LaTeX lint)

> GPU/CUDA is optional. If you need GPU acceleration, install the matching PyTorch build following the official PyTorch instructions.

### 1) Install (conda recommended)

```bash
conda create -n xscientist python=3.11 -y
conda activate xscientist

pip install -r requirements.txt
```

More reproducible (CI-style) install (optional):

```bash
pip install -r requirements.txt -c constraints-ci.txt
```

### 2) Configure API keys (as needed)

Set the env vars for your provider(s) (you do not need all of them):

```bash
export OPENAI_API_KEY="..."
export ZHIPU_API_KEY="..."
export GEMINI_API_KEY="..."
export S2_API_KEY="..."
```

### 3) Login (required)

```bash
python3 auth_cli.py login --user <your_name>
python3 auth_cli.py status
```

Login guard doc: `docs/LOGIN_GUARDRAIL.md`

### 4) Preflight (recommended)

```bash
python3 preflight_check.py --strict
python3 validate_repo.py
make smoke
```

---

## Configuration

### Output directory (do not write into the repo by default)

To keep the repo clean, outputs are written to a sibling directory by default:

- Default output root: sibling `<repo-name>_outputs`; for this repo that is `../XScientist_outputs`
- Priority: `RESEARCH_OUTPUT_DIR` > `AI_SCIENTIST_OUTPUT_DIR` > default sibling dir
- Fallback: if the sibling dir is not writable, use a system data dir (e.g., `~/.local/share/ai_scientist/research`)

Recommended: set an explicit output root.

```bash
export RESEARCH_OUTPUT_DIR="/path/to/my_xscientist_outputs"
```

### Strict fallback policy (debugging note)

Most scripts support stricter quality gates. During local debugging you may choose to relax strict fallbacks via `--override-strict-fallbacks` (not recommended for serious runs).

---

## Usage

### A) Run a single project from a topic

```bash
python3 run_project.py my_project \
  --output-root "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md
```

More usage: `docs/guides/PROJECT_USAGE.md`

### B) Continuous/batch generation

```bash
python3 continuous_paper_generator.py \
  --research-dir "$RESEARCH_OUTPUT_DIR" \
  --topic examples/example_topic.md \
  --paper-types icbinb
```

### C) Long-running daemon (recommended for continuous iteration)

```bash
python3 continuous_research_daemon.py \
  --source-config configs/sources/stable_source_priority.example.json \
  --duration-hours 24 \
  --enable-rewrite-followup \
  --auto-source-quality-feedback \
  --auto-quality-strategy-feedback \
  --auto-quality-governor \
  --auto-evidence-strategy-feedback \
  --auto-export-submission-dossier \
  --auto-failure-guard \
  --serve-dashboard \
  -- --submission-mode --num-ideas 3
```

Common ops commands:

```bash
bash run_stable_daemon.sh status
bash run_stable_daemon.sh brief
bash run_stable_daemon.sh handoff
bash run_stable_daemon.sh report-trends
bash run_stable_daemon.sh source-plan
```

### D) Feedback system monitoring

```bash
# Check system health
python3 feedback_cli.py --feedback-dir ./feedback status

# View recommended actions
python3 feedback_cli.py --feedback-dir ./feedback actions

# Analyze trends
python3 feedback_cli.py --feedback-dir ./feedback trends \
  --metrics quality_score success_rate error_rate

# Export report
python3 feedback_cli.py --feedback-dir ./feedback report
```

More usage: `docs/guides/FEEDBACK_QUICKSTART.md`

---

## Outputs & Observability

XScientist writes structured artifacts under the output root (directory names may evolve across versions):

- `projects/`: full per-project directories
- `experiments/`: experiment outputs and logs
- `ideas/`: idea artifacts
- `papers/`: per-paper directories from batch generation
- `batches/`: continuous-generator batch progress and reports
- `cache/`: HuggingFace / Torch / wandb runtime caches
- `reports/`: trends/handoff reports (daemon)
- `knowledge_base/`: cross-project memory (e.g., self-evolution history/playbook)

Common index/board commands (see `research_manager.py --help` for more):

```bash
python3 research_manager.py rebuild-index
python3 research_manager.py submission-board --top 5 --require-gate
python3 research_manager.py rewrite-board --top 10
python3 research_manager.py repair-board --top 20 --priority-tier p0
python3 research_manager.py evolution-board --top 20
python3 research_manager.py process-board --status blocked --top 30
```

---

## Example Papers

Example papers and related submission artifacts are collected in `example/` for checking paper formatting, supplementary material organization, and final delivery structure.

Currently organized example files:

- [example/XScientist_Board.pdf](example/XScientist_Board.pdf): XScientist Board paper/report PDF.
- [example/icml_submitted_gravitation_paper.pdf](example/icml_submitted_gravitation_paper.pdf): ICML-submitted gravitation manuscript PDF.

---

## Docs

- `docs/guides/PROJECT_USAGE.md`: `run_project.py` usage and flags
- `docs/guides/FEEDBACK_QUICKSTART.md`: Feedback system quick start guide
- `docs/CONFIG_REFERENCE.md`: detailed configuration and parameters
- `docs/SOURCE_ORCHESTRATION.md`: source queue orchestration and recommended run postures
- `docs/LONG_RUNNING_GUIDE.md`: Long-running operations guide
- `docs/LOGIN_GUARDRAIL.md`: login guard and session management
- `docs/guides/OUTPUT_DIRECTORIES.md`: output directory policy (if it diverges from code, follow `ai_scientist/config/paths.py`)
- `ARCHITECTURE.md`: System architecture documentation
- `OPTIMIZATION_SUMMARY.md`: Optimization summary

---

## Development

- Unit tests: `make test`
- Syntax/import/validation smoke: `make smoke`
- Stricter local doctor: `make doctor` (requires a valid login session)
- Formatting: `make format`

---

## Roadmap

XScientist's roadmap focuses on moving autonomous research from "one-shot paper generation" toward long-running, reproducible, reviewable, and submission-ready research infrastructure. Issues and PRs are welcome (see `CONTRIBUTING.md`).

**Near Term: Stable Delivery and Reproducibility**

- [ ] Complete the `example/` paper and supplementary-material set as submission-ready reference outputs.
- [ ] Expand preflight / smoke checks for API keys, LaTeX, output directories, login state, and dependency versions.
- [ ] Add a paper delivery checklist covering PDFs, figures, tables, citations, experiment logs, and reproducibility configs.
- [ ] Integrate TODO closure signals into quality gates so each paper exposes unresolved experiments and evidence gaps.

**Mid Term: Self-Review, Repair, and Quality Improvement**

- [ ] Strengthen bidirectional binding between evidence metrics and experiment outputs (figures, tables, and metrics).
- [ ] Add consistency and regression checks for submission-ready dossiers.
- [ ] Wire self-evolution / playbook signals more directly into automatic rewrite, repair, and follow-up executors.
- [ ] Add multi-reviewer aggregation across novelty, soundness, clarity, reproducibility, and ethics risks.

**Long Term: Continuous Autonomous Research**

- [ ] Let the daemon adapt research strategy from historical success rate, cost, quality scores, and failure modes.
- [ ] Build a cross-project knowledge base for strong ideas, failure cases, reusable experiment templates, and writing lessons.
- [ ] Provide more complete English docs, API docs, and plugin interfaces for external collaboration and extension.
- [ ] Support standard benchmarks / leaderboards for evaluating long-running autonomous research systems.

---

## System Architecture

For detailed architecture documentation, see: [ARCHITECTURE.md](ARCHITECTURE.md)

Core components:
- **Ideation Engine**: Idea generation and ranking
- **Experiments Engine**: Experiment execution and evidence collection
- **Writeup Engine**: Paper writing and compilation
- **Self-Review Engine**: Self-review and repair
- **Autonomous Evolution Engine**: Autonomous evolution and strategy optimization
- **Adaptive Learning Engine**: Adaptive learning and recommendations
- **Enhanced Feedback System**: Enhanced feedback and monitoring

## Contributing & Community

- Contributing guide: `CONTRIBUTING.md`
- Code of conduct: `CODE_OF_CONDUCT.md`
- Security policy: `SECURITY.md`
- Architecture docs: `ARCHITECTURE.md`

---

## License

Apache-2.0. See `LICENSE`.

---

## Acknowledgements

Thanks to the open-source projects that inspired parts of this work:

- [Sakana AI: AI Scientist](https://github.com/SakanaAI/AI-Scientist)
- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- [awesome-ai-research-writing](https://github.com/Leey21/awesome-ai-research-writing)
- [AIDE](https://github.com/WecoAI/aideml)
- [DeepReviewer-v2](https://github.com/ResearAI/DeepReviewer-v2)

---

## Citation and References

If you use XScientist in research, please cite this project and the generated paper you used. For papers or reports, include the commit hash, experiment configuration, model versions, and output directory for reproducibility.

### XScientist

XScientist (software / repository):

```bibtex
@software{xscientist,
  title        = {XScientist: A Long-Running Autonomous Scientific Research System},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/XScientist}
}
```

XScientist Board (paper or report authored/refined with this system):

```bibtex
@misc{xscientist_board,
  title        = {XScientist Board: Artifact-Routed Submission Hardening for Autonomous Research Systems},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/XScientist/blob/main/example/XScientist_Board.pdf}
}
```

ICML-submitted gravitation example paper:

```bibtex
@misc{xscientist_icml_submitted_gravitation,
  title        = {A Gravitational Field Theory for Deep Networks},
  author       = {{XScientist}},
  year         = {2026},
  url          = {https://github.com/smileformylove/XScientist/blob/main/example/icml_submitted_gravitation_paper.pdf}
}
```

### Citation Notes

- When citing papers generated by XScientist, cite both this repository and the specific generated paper.
- Clearly describe any human review, filtering, rewriting, or post-processing applied to generated results.
